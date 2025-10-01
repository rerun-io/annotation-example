from dataclasses import dataclass, field
from pathlib import Path
from timeit import default_timer as timer
from typing import Literal

import numpy as np
import rerun as rr
from jaxopt._src.levenberg_marquardt import LevenbergMarquardtState
from jaxtyping import Bool, Float, Float32, Int, UInt8
from numpy import ndarray
from simplecv.apis.view_exoego import compute_vertex_normals_batch
from simplecv.camera_parameters import Intrinsics, PinholeParameters
from simplecv.data.skeleton.coco_133 import LEFT_HAND_IDX, RIGHT_HAND_IDX
from simplecv.data.skeleton.mediapipe import MEDIAPIPE_IDS
from simplecv.ops.mano.mano_np import ManoSimpleLayerNP
from simplecv.ops.mano.optim_jax_single import OptimInput, OptimResult, PoseOptimConfig, SingleHandOptim
from simplecv.ops.triangulate import batch_triangulate
from simplecv.rerun_log_utils import (
    Points2DWithConfidence,
    Points3DWithConfidence,
    confidence_scores_to_rgb,
)
from wilor_nano.hand_detection import DetectionResult, HandDetector
from wilor_nano.hand_keypoints import KeypointResults, WilorHandKeypointDetector

from annotation_example.hand_calibrator import align_rotation
from annotation_example.one_euro_filter import OneEuroFilter

HandLabel = Literal["left", "right"]
HAND_LABELS: tuple[HandLabel, HandLabel] = ("left", "right")


@dataclass
class ManoResults:
    """Container for MANO fitting results"""

    global_orient: Float32[ndarray, "3"]
    """Global orientation of the hand in axis-angle format."""
    hand_pose: Float32[ndarray, "45"]
    """Pose parameters for the hand."""
    betas: Float32[ndarray, "10"]
    """Shape parameters for the hand."""
    translation: Float32[ndarray, "3"]
    """Translation of the hand in 3D space."""


@dataclass
class ManoHistory:
    """Cache of the MANO fit for the current frame and two predecessors."""

    t_mano: ManoResults | None = None
    """MANO fit for the current frame (t); ``None`` when unavailable."""
    t_minus_1_mano: ManoResults | None = None
    """MANO fit from the previous frame (t-1); ``None`` when unavailable."""


@dataclass
class MultiHandState:
    """Rolling MANO histories for both hands tracked across frames."""

    left: ManoHistory = field(default_factory=ManoHistory)
    """Temporal MANO history for the left hand."""
    right: ManoHistory = field(default_factory=ManoHistory)
    """Temporal MANO history for the right hand."""


@dataclass
class MultiViewHandTrackerConfig:
    """Configuration for coordinating multi-view detections and keypoints."""

    detection_confidence: float = 0.6
    """Confidence threshold for the hand detector."""
    keypoint_confidence: float = 0.35
    """Minimum mean confidence for per-hand keypoints before zeroing detections."""
    verbose: bool = False
    """Whether to log verbose information."""
    dbt: bool = False
    """Toggle detection-by-tracking extrapolation; fallback to raw detections when disabled."""


class MultiViewHandTracker:
    """Wraps the multi-view detection and keypoint pipeline for calibrated rigs. Assumes a known hand shape"""

    def __init__(
        self,
        config: MultiViewHandTrackerConfig,
        hand_detector: HandDetector,
        hand_keypoint_detector: WilorHandKeypointDetector,
        betas: Float32[ndarray, "10"],
        pinhole_param_list: list[PinholeParameters],
        parent_log_path: Path,
    ) -> None:
        """Persist the tracker configuration for later inference runs."""
        self.config: MultiViewHandTrackerConfig = config
        self.hand_detector: HandDetector = hand_detector
        self.hand_keypoint_detector: WilorHandKeypointDetector = hand_keypoint_detector
        self.betas: Float32[ndarray, "10"] = betas
        self.parent_log_path: Path = parent_log_path
        self.Pall_exo: Float[ndarray, "n_views 3 4"] = np.stack(
            [pinhole.projection_matrix for pinhole in pinhole_param_list]
        )

        self.left_mano_layer: ManoSimpleLayerNP = ManoSimpleLayerNP(side="left", mano_root=Path("data/"))
        self.right_mano_layer: ManoSimpleLayerNP = ManoSimpleLayerNP(side="right", mano_root=Path("data/"))

        left_optim_cfg = PoseOptimConfig(
            beta=self.betas,
            Pall=self.Pall_exo,
            hand_side="left",
        )
        right_optim_cfg = PoseOptimConfig(
            beta=self.betas,
            Pall=self.Pall_exo,
            hand_side="right",
        )
        self.left_hand_optimizer: SingleHandOptim = SingleHandOptim(config=left_optim_cfg)
        self.right_hand_optimizer: SingleHandOptim = SingleHandOptim(config=right_optim_cfg)

        self.left_bbox_filter: OneEuroFilter | None = None
        self.right_bbox_filter: OneEuroFilter | None = None
        self.left_bbox_time: float = 0.0
        self.right_bbox_time: float = 0.0

    def __call__(
        self,
        *,
        rgb_batch: UInt8[ndarray, "n_views H W 3"],
        pinhole_param_list: list[PinholeParameters],
        hand_state: MultiHandState,
        recording: rr.RecordingStream | None = None,
    ) -> MultiHandState:
        """Run hand detection followed by per-view keypoint refinement."""

        for hand_label in HAND_LABELS:
            mano_history: ManoHistory = getattr(hand_state, hand_label)
            use_tracking: bool = (
                self.config.dbt and mano_history.t_mano is not None and mano_history.t_minus_1_mano is not None
            )
            if not use_tracking:
                xyxy_batch: Float[ndarray, "n_views 1 4"] = self._detect_hands(
                    rgb_batch=rgb_batch,
                    pinhole_param_list=pinhole_param_list,
                    recording=recording,
                    hand_label=hand_label,
                )
            else:
                xyxy_batch: Float[ndarray, "n_views 1 4"] = self._detection_by_tracking(
                    rgb_batch=rgb_batch,
                    pinhole_param_list=pinhole_param_list,
                    mano_history=mano_history,
                    recording=recording,
                    hand_label=hand_label,
                )

            uvc_batch: Float[ndarray, "n_views mp_kpts=21 3"] = self._detect_hand_keypoints(
                rgb_batch=rgb_batch,
                xyxy_batch=xyxy_batch,
                pinhole_param_list=pinhole_param_list,
                recording=recording,
                hand_label=hand_label,
            )

            finite_uv_mask: Bool[ndarray, "n_views mp_kpts=21"] = np.isfinite(uvc_batch[:, :, 0]) & np.isfinite(
                uvc_batch[:, :, 1]
            )
            confident_mask: Bool[ndarray, "n_views mp_kpts=21"] = uvc_batch[:, :, 2] > 0.0  # type: ignore Pyrefly bug
            valid_keypoint_mask: Bool[ndarray, "n_views mp_kpts=21"] = finite_uv_mask & confident_mask
            has_valid_keypoints: bool = bool(np.any(valid_keypoint_mask))
            if not has_valid_keypoints:
                mano_history.t_mano = None
                mano_history.t_minus_1_mano = None
                continue

            xyzc: Float[ndarray, "mp_kpts=21 4"] | None = self._triangulate_keypoints(
                hand_uvc=uvc_batch,
                pinhole_param_list=pinhole_param_list,
                hand_label=hand_label,
                recording=recording,
            )

            mano_fit: ManoResults | None = self._fit_mano_model(
                uvc_batch=uvc_batch,
                xyzc=xyzc,
                prev_mano=mano_history.t_mano,
                hand_label=hand_label,
            )

            if mano_fit is not None:
                mano_history.t_minus_1_mano = mano_history.t_mano
                mano_history.t_mano = mano_fit
            else:
                mano_history.t_minus_1_mano = None
                mano_history.t_mano = None

        return hand_state

    def _detect_hands(
        self,
        *,
        rgb_batch: UInt8[ndarray, "n_views H W 3"],
        pinhole_param_list: list[PinholeParameters],
        hand_label: HandLabel,
        recording: rr.RecordingStream | None = None,
    ) -> Float[ndarray, "n_views 1 4"]:
        xyxy_list: list[Float[ndarray, "1 4"]] = []

        for pinhole_param, rgb_hw3 in zip(pinhole_param_list, rgb_batch, strict=True):
            det_result: DetectionResult = self.hand_detector(
                rgb_hw3=rgb_hw3,
                hand_conf=self.config.detection_confidence,
            )
            raw_xyxy: Float[ndarray, "1 4"] | None = (
                det_result.left_xyxy if hand_label == "left" else det_result.right_xyxy
            )
            xyxy: Float[ndarray, "1 4"] = (
                raw_xyxy if raw_xyxy is not None else np.full((1, 4), np.nan, dtype=np.float32)
            )
            xyxy_list.append(xyxy)
            if self.config.verbose:
                cam_log_path: Path = self.parent_log_path / "exo" / pinhole_param.name
                pinhole_log_path: Path = cam_log_path / "pinhole"
                rr.log(
                    f"{pinhole_log_path}/{hand_label}_hand_bbox",
                    rr.Boxes2D(
                        array=xyxy,
                        array_format=rr.Box2DFormat.XYXY,
                        class_ids=0 if hand_label == "left" else 1,
                    ),
                    recording=recording,
                )

        xyxy_batch: Float[ndarray, "n_views 1 4"] = np.stack(xyxy_list, axis=0)

        return xyxy_batch

    def _detect_hand_keypoints(
        self,
        *,
        rgb_batch: UInt8[ndarray, "n_views H W 3"],
        xyxy_batch: Float[ndarray, "n_views 1 4"],
        pinhole_param_list: list[PinholeParameters],
        hand_label: HandLabel,
        recording: rr.RecordingStream | None = None,
    ) -> Float[ndarray, "n_views mp_kpts=21 3"]:
        """Detect Mediapipe keypoints for a single hand across all views."""

        n_views: int = int(rgb_batch.shape[0])
        hand_uvc: Float[ndarray, "n_views mp_kpts=21 3"] = np.full((n_views, 21, 3), np.nan, dtype=np.float32)

        for view_idx, (rgb_hw3, bbox_xyxy, pinhole_param) in enumerate(
            zip(rgb_batch, xyxy_batch, pinhole_param_list, strict=True)
        ):
            if np.isnan(bbox_xyxy).any():
                continue

            kpts_results: KeypointResults = self.hand_keypoint_detector(
                rgb_hw3=rgb_hw3,
                xyxy=bbox_xyxy,
                handedness=hand_label,
            )
            uv: Float[ndarray, "n_frames=1 mp_kpts=21 2"] = kpts_results.keypoints_2d
            conf: Float[ndarray, "n_frames=1 mp_kpts=21"] = kpts_results.scores
            uv_21x2: Float[ndarray, "mp_kpts=21 2"] = uv[0].astype(np.float32, copy=True)
            conf_values: Float[ndarray, "mp_kpts=21"] = conf[0].astype(np.float32, copy=True)

            mean_conf: float = float(np.nanmean(conf_values))
            if not np.isfinite(mean_conf) or mean_conf < self.config.keypoint_confidence:
                uv_21x2[:] = np.nan
                conf_values[:] = 0.0

            hand_uvc[view_idx, :, :2] = uv_21x2
            hand_uvc[view_idx, :, 2] = conf_values

            if self.config.verbose:
                camera_name: str = getattr(pinhole_param, "name", f"camera_{view_idx}")
                hand_log_path: Path = self.parent_log_path / "exo" / camera_name / "pinhole" / "video" / hand_label
                conf_colors: UInt8[ndarray, "1 mp_kpts=21 3"] = confidence_scores_to_rgb(
                    confidence_scores=conf_values[np.newaxis, :, np.newaxis]
                )
                rr.log(
                    f"{hand_log_path}/keypoints",
                    Points2DWithConfidence(
                        positions=uv_21x2,
                        confidences=conf_values,
                        class_ids=0 if hand_label == "left" else 1,
                        keypoint_ids=MEDIAPIPE_IDS,
                        show_labels=False,
                        colors=conf_colors[0],
                    ),
                    recording=recording,
                )

        return hand_uvc

    def _triangulate_keypoints(
        self,
        *,
        hand_uvc: Float[ndarray, "n_views mp_kpts=21 3"],
        pinhole_param_list: list[PinholeParameters],
        hand_label: HandLabel,
        recording: rr.RecordingStream | None = None,
    ) -> Float[ndarray, "mp_kpts=21 4"] | None:
        """Triangulate a single hand's keypoints into world coordinates."""

        view_support: ndarray = np.sum(~np.isnan(hand_uvc[:, :, 0]), axis=0)
        max_views: int = int(view_support.max()) if view_support.size > 0 else 0
        if max_views < 2:
            return None

        hand_uvc_f32: Float32[ndarray, "n_views mp_kpts=21 3"] = hand_uvc.astype(np.float32, copy=True)
        uvc_for_triangulation: Float32[ndarray, "n_views mp_kpts=21 3"] = np.nan_to_num(hand_uvc_f32, nan=0.0)
        projection_stack: Float32[ndarray, "n_views 3 4"] = np.stack(
            [pinhole.projection_matrix for pinhole in pinhole_param_list]
        ).astype(np.float32)

        xyzc_raw: ndarray = batch_triangulate(
            uvc_for_triangulation,
            projection_stack,
            min_views=2,
        )
        xyzc: Float32[ndarray, "mp_kpts=21 4"] = xyzc_raw.astype(np.float32, copy=False)
        confidences: Float32[ndarray, "mp_kpts=21"] = xyzc[:, 3]
        invalid_mask: ndarray = confidences <= 0.0  # type: ignore Pyrefly bug

        if np.any(invalid_mask):
            xyzc[invalid_mask, :3] = np.nan

        if self.config.verbose:
            hand_log_path: Path = self.parent_log_path / "triangulated" / hand_label
            rr.log(
                str(hand_log_path),
                Points3DWithConfidence(
                    positions=xyzc[:, :3],
                    confidences=xyzc[:, 3],
                    class_ids=0 if hand_label == "left" else 1,
                    keypoint_ids=MEDIAPIPE_IDS,
                    show_labels=False,
                    colors=confidence_scores_to_rgb(confidences[np.newaxis, :, np.newaxis])[0],
                ),
                recording=recording,
            )

        return xyzc

    def _fit_mano_model(
        self,
        *,
        uvc_batch: Float[ndarray, "n_views mp_kpts=21 3"],
        xyzc: Float[ndarray, "mp_kpts=21 4"] | None,
        prev_mano: ManoResults | None,
        hand_label: HandLabel,
    ) -> ManoResults | None:
        """Fit the MANO model to the 2D multi-view keypoints."""
        # start by initializing from previous fit if available, otherwise from 3D keypoints if available,
        # otherwise from a sane default if only a single view is available.
        mano_init: ManoResults = self._initialize_mano_params(
            prev_mano=prev_mano,
            xyzc=xyzc,
            hand_side=hand_label,
        )
        optimizer: SingleHandOptim = self.left_hand_optimizer if hand_label == "left" else self.right_hand_optimizer

        n_views: int = int(uvc_batch.shape[0])
        uv_only: Float32[ndarray, "n_views mp_kpts=21 2"] = uvc_batch[:, :, :2].astype(np.float32, copy=True)
        confidences_2d: Float32[ndarray, "n_views mp_kpts=21"] = uvc_batch[:, :, 2].astype(np.float32, copy=True)
        finite_mask: Bool[ndarray, "n_views mp_kpts=21"] = np.isfinite(uv_only[..., 0]) & np.isfinite(uv_only[..., 1])
        confident_mask: Bool[ndarray, "n_views mp_kpts=21"] = confidences_2d > 0.0  # type: ignore Pyrefly bug
        valid_mask: Bool[ndarray, "n_views mp_kpts=21"] = finite_mask & confident_mask
        if not np.any(valid_mask):
            return None
        uv_only[~valid_mask] = np.nan

        uv_pred_full: Float32[ndarray, "1 n_views 133 2"] = np.full(
            (1, n_views, 133, 2),
            np.nan,
            dtype=np.float32,
        )
        fill_idx = RIGHT_HAND_IDX if hand_label == "right" else LEFT_HAND_IDX
        for view_idx in range(n_views):
            uv_pred_full[0, view_idx, fill_idx, :] = uv_only[view_idx]
        so3_components: Float32[ndarray, "48"] = np.concatenate(
            [
                mano_init.global_orient.astype(np.float32, copy=False),
                mano_init.hand_pose.astype(np.float32, copy=False),
            ],
            axis=0,
        )
        so3_init: Float32[ndarray, "1 48"] = so3_components[np.newaxis, :]
        trans_init: Float32[ndarray, "1 3"] = mano_init.translation.astype(np.float32, copy=False)[np.newaxis, :]
        optim_input: OptimInput = OptimInput(uv_pred=uv_pred_full, so3_init=so3_init, trans_init=trans_init)
        try:
            optim_tuple: tuple[OptimResult, LevenbergMarquardtState] = optimizer(optim_input)
        except Exception:
            return None
        optim_result: OptimResult = optim_tuple[0]
        optim_state: LevenbergMarquardtState = optim_tuple[1]

        error_value: float = float(np.asarray(optim_state.error))
        if not np.isfinite(error_value):
            return None

        so3_optim: Float32[ndarray, "1 48"] = optim_result.so3_optim.astype(np.float32, copy=False)
        trans_optim: Float32[ndarray, "1 3"] = optim_result.trans_optim.astype(np.float32, copy=False)
        global_orient_optim: Float32[ndarray, "3"] = so3_optim[0, 0:3].astype(np.float32, copy=False)
        hand_pose_optim: Float32[ndarray, "45"] = so3_optim[0, 3:].astype(np.float32, copy=False)
        translation_optim: Float32[ndarray, "3"] = trans_optim[0].astype(np.float32, copy=False)
        betas_f32: Float32[ndarray, "10"] = self.betas.astype(np.float32, copy=False)

        mano_fit: ManoResults = ManoResults(
            global_orient=global_orient_optim,
            hand_pose=hand_pose_optim,
            betas=betas_f32,
            translation=translation_optim,
        )

        return mano_fit

    def _initialize_mano_params(
        self,
        *,
        prev_mano: ManoResults | None,
        xyzc: Float[ndarray, "mp_kpts=21 4"] | None,
        hand_side: HandLabel,
    ) -> ManoResults:
        """Return previous MANO fit when available; otherwise compute a fresh initialization."""
        if prev_mano is not None:
            # Reuse the prior frame's solution when available.
            return prev_mano

        mano_layer: ManoSimpleLayerNP = self.left_mano_layer if hand_side == "left" else self.right_mano_layer

        # Neutral pose + depth bias serve as an absolute fallback when no observations exist.
        default_global_orient: Float32[ndarray, "3"] = np.zeros((3,), dtype=np.float32)
        default_hand_pose: Float32[ndarray, "45"] = np.zeros((45,), dtype=np.float32)
        default_translation: Float32[ndarray, "3"] = np.array([0.0, 0.0, 0.6], dtype=np.float32)

        if xyzc is not None:
            hand_xyz: Float32[ndarray, "mp_kpts=21 3"] = xyzc[:, :3].astype(np.float32, copy=True)
            confidences: Float32[ndarray, "mp_kpts=21"] = xyzc[:, 3].astype(np.float32, copy=True)
            valid_mask: Bool[ndarray, "mp_kpts=21"] = np.isfinite(hand_xyz).all(axis=1) & (confidences > 0.0)

            if np.any(valid_mask):
                # Seed the wrist translation from the triangulated joint.
                translation_guess: Float32[ndarray, "3"] = (
                    hand_xyz[valid_mask]
                    .mean(axis=0)
                    .astype(
                        np.float32,
                        copy=False,
                    )
                )

                so3_naive: Float32[ndarray, "1 48"] = np.zeros((1, 48), dtype=np.float32)
                trans_naive: Float32[ndarray, "1 3"] = translation_guess[np.newaxis, :]
                betas_batch: Float32[ndarray, "1 10"] = np.broadcast_to(
                    self.betas.astype(np.float32, copy=False),
                    (1, self.betas.shape[0]),
                ).copy()

                # Synthesize a MANO mesh using the neutral pose for comparison purposes.
                mano_naive_results: tuple[
                    Float32[ndarray, "1 n_verts=778 3"],
                    Float32[ndarray, "1 joints=21 3"],
                ] = mano_layer(so3_naive, betas_batch, trans_naive)
                verts_naive_m: Float32[ndarray, "n_verts=778 3"] = (mano_naive_results[0][0] / 1000.0).astype(
                    np.float32, copy=False
                )
                joints_naive_m: Float32[ndarray, "joints=21 3"] = (mano_naive_results[1][0] / 1000.0).astype(
                    np.float32, copy=False
                )

                hand_xyz_masked: Float32[ndarray, "mp_kpts=21 3"] = hand_xyz.copy()
                hand_xyz_masked[~valid_mask] = np.nan

                # Align the MANO root orientation with the triangulated MCP directions.
                rotation_vec: Float32[ndarray, "3"] = align_rotation(joints_naive_m, hand_xyz_masked)
                so3_aligned: Float32[ndarray, "1 48"] = so3_naive.copy()
                so3_aligned[0, 0:3] = rotation_vec.astype(np.float32, copy=False)

                # Evaluate the aligned rotation before applying any translation correction.
                mano_aligned_pre: tuple[
                    Float32[ndarray, "1 n_verts=778 3"],
                    Float32[ndarray, "1 joints=21 3"],
                ] = mano_layer(so3_aligned, betas_batch, trans_naive)
                joints_aligned_pre_m: Float32[ndarray, "joints=21 3"] = (mano_aligned_pre[1][0] / 1000.0).astype(
                    np.float32, copy=False
                )

                # Adjust the translation so aligned joints sit on the triangulated targets.
                translation_offset: Float32[ndarray, "3"] = (
                    (hand_xyz[valid_mask] - joints_aligned_pre_m[valid_mask])
                    .mean(axis=0)
                    .astype(np.float32, copy=False)
                )

                trans_aligned: Float32[ndarray, "1 3"] = trans_naive + translation_offset[np.newaxis, :]

                mano_aligned_results: tuple[
                    Float32[ndarray, "1 n_verts=778 3"],
                    Float32[ndarray, "1 joints=21 3"],
                ] = mano_layer(so3_aligned, betas_batch, trans_aligned)
                verts_aligned_m: Float32[ndarray, "n_verts=778 3"] = (mano_aligned_results[0][0] / 1000.0).astype(
                    np.float32, copy=False
                )

                global_orient_init: Float32[ndarray, "3"] = so3_aligned[0, 0:3].astype(np.float32, copy=False)
                hand_pose_init: Float32[ndarray, "45"] = so3_aligned[0, 3:].astype(np.float32, copy=False)
                translation_init: Float32[ndarray, "3"] = trans_aligned[0].astype(np.float32, copy=False)

                if self.config.verbose:
                    # Visual sanity checks of naive vs aligned initial meshes.
                    faces_np: Int[ndarray, "n_faces=1538 3"] = mano_layer.th_faces.astype(np.int32, copy=False)
                    normals_naive: Float32[ndarray, "n_verts=778 3"] = compute_vertex_normals_batch(
                        verts_naive_m[np.newaxis, ...],
                        faces_np,
                    )[0].astype(np.float32, copy=False)
                    normals_aligned: Float32[ndarray, "n_verts=778 3"] = compute_vertex_normals_batch(
                        verts_aligned_m[np.newaxis, ...],
                        faces_np,
                    )[0].astype(np.float32, copy=False)
                    mano_mesh_path: Path = self.parent_log_path / "mano_init" / hand_side
                    rr.log(
                        f"{mano_mesh_path}_init",
                        rr.Mesh3D(
                            vertex_positions=verts_naive_m,
                            triangle_indices=faces_np,
                            vertex_normals=normals_naive,
                            albedo_factor=(255, 64, 0, 255),
                        ),
                    )
                    rr.log(
                        f"{mano_mesh_path}_aligned",
                        rr.Mesh3D(
                            vertex_positions=verts_aligned_m,
                            triangle_indices=faces_np,
                            vertex_normals=normals_aligned,
                            albedo_factor=(0, 255, 0, 255),
                        ),
                    )

                return ManoResults(
                    global_orient=global_orient_init,
                    hand_pose=hand_pose_init,
                    betas=self.betas.astype(np.float32, copy=False),
                    translation=translation_init,
                )

        return ManoResults(
            # Default neutral pose gives downstream optimizers a deterministic fallback.
            global_orient=default_global_orient,
            hand_pose=default_hand_pose,
            betas=self.betas.astype(np.float32, copy=False),
            translation=default_translation,
        )

    def _detection_by_tracking(
        self,
        *,
        rgb_batch: UInt8[ndarray, "n_views H W 3"],
        pinhole_param_list: list[PinholeParameters],
        mano_history: ManoHistory,
        recording: rr.RecordingStream | None,
        hand_label: HandLabel,
    ) -> Float32[ndarray, "n_views 1 4"]:
        """Propagate prior MANO fits to synthesize per-view detections."""

        prev_mano: ManoResults | None = mano_history.t_mano
        prev_prev_mano: ManoResults | None = mano_history.t_minus_1_mano
        if prev_mano is None or prev_prev_mano is None:
            return np.full((len(pinhole_param_list), 1, 4), np.nan, dtype=np.float32)

        global_orient_prev: Float32[ndarray, "3"] = prev_mano.global_orient.astype(np.float32, copy=False)
        global_orient_prev_prev: Float32[ndarray, "3"] = prev_prev_mano.global_orient.astype(np.float32, copy=False)
        hand_pose_prev: Float32[ndarray, "45"] = prev_mano.hand_pose.astype(np.float32, copy=False)
        hand_pose_prev_prev: Float32[ndarray, "45"] = prev_prev_mano.hand_pose.astype(np.float32, copy=False)
        translation_prev: Float32[ndarray, "3"] = prev_mano.translation.astype(np.float32, copy=False)
        translation_prev_prev: Float32[ndarray, "3"] = prev_prev_mano.translation.astype(np.float32, copy=False)
        betas_prev: Float32[ndarray, "10"] = prev_mano.betas.astype(np.float32, copy=False)

        two_scalar: float = float(np.float32(2.0))
        global_orient_hat: Float32[ndarray, "3"] = ((two_scalar * global_orient_prev) - global_orient_prev_prev).astype(
            np.float32, copy=False
        )
        hand_pose_hat: Float32[ndarray, "45"] = ((two_scalar * hand_pose_prev) - hand_pose_prev_prev).astype(
            np.float32, copy=False
        )
        translation_hat: Float32[ndarray, "3"] = ((two_scalar * translation_prev) - translation_prev_prev).astype(
            np.float32, copy=False
        )

        so3_hat_components: Float32[ndarray, "48"] = np.concatenate(
            [global_orient_hat, hand_pose_hat],
            axis=0,
        ).astype(np.float32, copy=False)
        so3_hat_batch: Float32[ndarray, "1 48"] = so3_hat_components[np.newaxis, :]
        betas_batch: Float32[ndarray, "1 10"] = betas_prev[np.newaxis, :]
        translation_batch: Float32[ndarray, "1 3"] = translation_hat[np.newaxis, :]

        mano_layer: ManoSimpleLayerNP = self.left_mano_layer if hand_label == "left" else self.right_mano_layer
        mano_forward: tuple[
            Float32[ndarray, "1 n_verts=778 3"],
            Float32[ndarray, "1 mp_kpts=21 3"],
        ] = mano_layer(
            th_pose_coeffs=so3_hat_batch,
            th_betas=betas_batch,
            th_trans=translation_batch,
        )
        joints_mm: Float32[ndarray, "mp_kpts=21 3"] = mano_forward[1][0].astype(np.float32, copy=False)
        joints_m: Float32[ndarray, "mp_kpts=21 3"] = (joints_mm / np.float32(1000.0)).astype(np.float32, copy=False)

        ones_column: Float32[ndarray, "mp_kpts=21 1"] = np.ones((joints_m.shape[0], 1), dtype=np.float32)
        world_keypoints_h: Float32[ndarray, "mp_kpts=21 4"] = np.concatenate(
            [joints_m, ones_column],
            axis=1,
        )

        xyxy_list: list[Float32[ndarray, "1 4"]] = []
        tracking_color: UInt8[ndarray, "1 4"] = np.array([[128, 255, 128, 255]], dtype=np.uint8)

        for view_idx, (rgb_hw3, pinhole_param) in enumerate(zip(rgb_batch, pinhole_param_list, strict=True)):
            projection_matrix: Float32[ndarray, "3 4"] = pinhole_param.projection_matrix.astype(np.float32, copy=False)
            pixel_h: Float32[ndarray, "mp_kpts=21 3"] = (projection_matrix @ world_keypoints_h.T).T.astype(
                np.float32, copy=False
            )
            depths: Float32[ndarray, "mp_kpts=21"] = pixel_h[:, 2]
            positive_depth_mask: Bool[ndarray, "mp_kpts=21"] = depths > 0.0

            uv: Float32[ndarray, "mp_kpts=21 2"] = np.full((world_keypoints_h.shape[0], 2), np.nan, dtype=np.float32)
            np.divide(
                pixel_h[:, :2],
                depths[:, np.newaxis],
                out=uv,
                where=positive_depth_mask[:, np.newaxis],
            )
            finite_uv_mask: Bool[ndarray, "mp_kpts=21"] = (
                np.isfinite(uv[:, 0]) & np.isfinite(uv[:, 1]) & positive_depth_mask
            )

            xyxy_view: Float32[ndarray, "1 4"]
            if not np.any(finite_uv_mask):
                xyxy_view = np.full((1, 4), np.nan, dtype=np.float32)
            else:
                valid_uv: Float32[ndarray, "m 2"] = uv[finite_uv_mask]
                min_xy: Float32[ndarray, "2"] = valid_uv.min(axis=0).astype(np.float32, copy=False)
                max_xy: Float32[ndarray, "2"] = valid_uv.max(axis=0).astype(np.float32, copy=False)
                width: float = float(max_xy[0] - min_xy[0])
                height: float = float(max_xy[1] - min_xy[1])
                side_length: float = max(width, height, 1.0)
                side_length *= 1.5
                center_xy: Float32[ndarray, "2"] = ((min_xy + max_xy) / np.float32(2.0)).astype(np.float32, copy=False)
                half_side: float = side_length / 2.0
                x1: float = float(center_xy[0] - half_side)
                y1: float = float(center_xy[1] - half_side)
                x2: float = float(center_xy[0] + half_side)
                y2: float = float(center_xy[1] + half_side)

                intrinsics: Intrinsics = pinhole_param.intrinsics
                image_height: float = float(intrinsics.height if intrinsics.height is not None else rgb_hw3.shape[0])
                image_width: float = float(intrinsics.width if intrinsics.width is not None else rgb_hw3.shape[1])

                x1 = float(np.clip(x1, 0.0, image_width))
                y1 = float(np.clip(y1, 0.0, image_height))
                x2 = float(np.clip(x2, 0.0, image_width))
                y2 = float(np.clip(y2, 0.0, image_height))

                if x1 > x2:
                    x1, x2 = x2, x1
                if y1 > y2:
                    y1, y2 = y2, y1

                xyxy_view = np.array([[x1, y1, x2, y2]], dtype=np.float32)

            xyxy_list.append(xyxy_view)

            if self.config.verbose:
                camera_name: str = getattr(pinhole_param, "name", f"camera_{view_idx}")
                cam_log_path: Path = self.parent_log_path / "exo" / camera_name
                pinhole_log_path: Path = cam_log_path / "pinhole"
                rr.log(
                    f"{pinhole_log_path}/{hand_label}_hand_bbox",
                    rr.Boxes2D(
                        array=xyxy_view,
                        array_format=rr.Box2DFormat.XYXY,
                        class_ids=0 if hand_label == "left" else 1,
                        colors=tracking_color,
                    ),
                    recording=recording,
                )

        xyxy_batch: Float32[ndarray, "n_views 1 4"] = np.stack(xyxy_list, axis=0).astype(np.float32, copy=False)

        bbox_filter: OneEuroFilter | None
        bbox_time: float
        if hand_label == "left":
            bbox_filter = self.left_bbox_filter
            bbox_time = self.left_bbox_time
        else:
            bbox_filter = self.right_bbox_filter
            bbox_time = self.right_bbox_time

        has_valid_bbox: bool = bool(np.isfinite(xyxy_batch).any())
        if has_valid_bbox:
            timestamp_now: float = timer()
            if bbox_filter is None:
                bbox_filter = OneEuroFilter(
                    t0=timestamp_now,
                    x0=xyxy_batch.copy(),
                    min_cutoff=1.0,
                    beta=0.0,
                    d_cutoff=1.0,
                )
                bbox_time = timestamp_now
            else:
                filtered_batch: Float32[ndarray, "n_views 1 4"] = bbox_filter(
                    timestamp_now,
                    xyxy_batch,
                )
                xyxy_batch = filtered_batch
                bbox_time = timestamp_now

        if hand_label == "left":
            self.left_bbox_filter = bbox_filter
            self.left_bbox_time = bbox_time
        else:
            self.right_bbox_filter = bbox_filter
            self.right_bbox_time = bbox_time

        return xyxy_batch
