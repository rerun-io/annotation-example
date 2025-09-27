from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import numpy as np
import rerun as rr
from jaxtyping import Bool, Float, Float32, UInt8
from numpy import ndarray
from simplecv.camera_parameters import PinholeParameters
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

    detection_confidence: float = 0.5
    """Confidence threshold for the hand detector."""
    keypoint_confidence: float = 0.3
    """Minimum mean confidence for per-hand keypoints before zeroing detections."""
    verbose: bool = True
    """Whether to log verbose information."""


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
            if mano_history.t_mano is None or mano_history.t_minus_1_mano is None:
                xyxy_batch: Float[ndarray, "n_views 1 4"] = self._detect_hands(
                    rgb_batch=rgb_batch,
                    pinhole_param_list=pinhole_param_list,
                    recording=recording,
                    hand_label=hand_label,
                )
            else:
                raise NotImplementedError("Tracking not yet implemented")

            uvc_batch: Float[ndarray, "n_views mp_kpts=21 3"] = self._detect_hand_keypoints(
                rgb_batch=rgb_batch,
                xyxy_batch=xyxy_batch,
                pinhole_param_list=pinhole_param_list,
                recording=recording,
                hand_label=hand_label,
            )

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
        mano_init: ManoResults | None = self._initialize_mano_params(
            prev_mano=prev_mano,
            xyzc=xyzc,
            hand_side=hand_label,
        )

        optimizer: SingleHandOptim = (
            self.left_hand_optimizer if hand_label == "left" else self.right_hand_optimizer
        )

        n_views: int = int(uvc_batch.shape[0])
        uv_only: Float32[ndarray, "n_views mp_kpts=21 2"] = uvc_batch[:, :, :2].astype(np.float32, copy=True)
        confidences_2d: Float32[ndarray, "n_views mp_kpts=21"] = uvc_batch[:, :, 2].astype(np.float32, copy=True)
        finite_mask: Bool[ndarray, "n_views mp_kpts=21"] = np.isfinite(uv_only[..., 0]) & np.isfinite(uv_only[..., 1])
        confident_mask: Bool[ndarray, "n_views mp_kpts=21"] = confidences_2d > 0.0
        valid_mask: Bool[ndarray, "n_views mp_kpts=21"] = finite_mask & confident_mask

        if not np.any(valid_mask):
            return prev_mano

        uv_only[~valid_mask] = np.nan

        uv_pred: Float32[ndarray, "1 n_views 133 2"] = np.full(
            (1, n_views, 133, 2),
            np.nan,
            dtype=np.float32,
        )
        hand_indices: ndarray = LEFT_HAND_IDX if hand_label == "left" else RIGHT_HAND_IDX
        uv_pred[0, :, hand_indices, :] = uv_only

        so3_init: Float32[ndarray, "1 48"]
        trans_init: Float32[ndarray, "1 3"]
        if mano_init is not None:
            so3_components: Float32[ndarray, "48"] = np.concatenate(
                [
                    mano_init.global_orient.astype(np.float32, copy=False),
                    mano_init.hand_pose.astype(np.float32, copy=False),
                ],
                axis=0,
            )
            so3_init = so3_components[np.newaxis, :]
            trans_init = mano_init.translation.astype(np.float32, copy=False)[np.newaxis, :]
        else:
            so3_init = np.zeros((1, 48), dtype=np.float32)
            trans_init = np.zeros((1, 3), dtype=np.float32)
            trans_init[0, 2] = 0.6

            if xyzc is not None:
                xyz_points: Float32[ndarray, "mp_kpts=21 3"] = xyzc[:, :3].astype(np.float32, copy=True)
                xyz_conf: Float32[ndarray, "mp_kpts=21"] = xyzc[:, 3].astype(np.float32, copy=True)
                xyz_valid_mask: Bool[ndarray, "mp_kpts=21"] = np.isfinite(xyz_points).all(axis=1) & (xyz_conf > 0.0)
                if np.any(xyz_valid_mask):
                    translation_estimate: Float32[ndarray, "3"] = np.nanmean(
                        xyz_points[xyz_valid_mask],
                        axis=0,
                    ).astype(np.float32, copy=False)
                    trans_init[0] = translation_estimate

        optim_input: OptimInput = OptimInput(
            uv_pred=uv_pred,
            so3_init=so3_init,
            trans_init=trans_init,
        )

        try:
            optim_result: OptimResult
            optim_result, _ = optimizer(optim_input)
        except Exception:
            return prev_mano

        so3_optim: Float32[ndarray, "1 48"] = optim_result.so3_optim.astype(np.float32, copy=True)
        trans_optim: Float32[ndarray, "1 3"] = optim_result.trans_optim.astype(np.float32, copy=True)
        so3_vector: Float32[ndarray, "48"] = so3_optim[0]
        global_orient: Float32[ndarray, "3"] = so3_vector[:3]
        hand_pose: Float32[ndarray, "45"] = so3_vector[3:]
        translation: Float32[ndarray, "3"] = trans_optim[0]

        if not np.isfinite(global_orient).all() or not np.isfinite(hand_pose).all() or not np.isfinite(translation).all():
            return prev_mano

        mano_results: ManoResults = ManoResults(
            global_orient=global_orient,
            hand_pose=hand_pose,
            betas=self.betas.astype(np.float32, copy=False),
            translation=translation,
        )

        return mano_results

    def _initialize_mano_params(
        self,
        *,
        prev_mano: ManoResults | None,
        xyzc: Float[ndarray, "mp_kpts=21 4"] | None,
        hand_side: HandLabel,
    ) -> ManoResults | None:
        """Return previous MANO fit when available; otherwise signal downstream initialization."""
        if prev_mano is not None:
            return prev_mano
        if xyzc is not None:
            return None
