from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import cv2
import numpy as np
import rerun as rr
from jaxopt._src.levenberg_marquardt import LevenbergMarquardtState
from jaxtyping import Bool, Float, Float32, Int, UInt8
from numpy import ndarray
from simplecv.apis.view_exoego import compute_vertex_normals_batch
from simplecv.camera_parameters import PinholeParameters
from simplecv.data.skeleton.coco_133 import COCO_133_IDS, LEFT_HAND_IDX, RIGHT_HAND_IDX
from simplecv.data.skeleton.mediapipe import MEDIAPIPE_IDS
from simplecv.ops.mano.mano_np import MANOLayerNP
from simplecv.ops.mano.optim_jax_single_shape import (
    LossWeights,
    OptimShapeInput,
    OptimShapeResult,
    PoseShapeOptimConfig,
    SingleHandShapeOptim,
)
from simplecv.ops.triangulate import batch_triangulate
from simplecv.rerun_log_utils import Points2DWithConfidence, Points3DWithConfidence, confidence_scores_to_rgb
from wilor_nano.hand_detection import DetectionResult, HandDetector
from wilor_nano.hand_keypoints import KeypointResults, WilorHandKeypointDetector


def align_rotation(
    xyz_mano: Float[ndarray, "n_kpts 3"], xyz_triangulated: Float[ndarray, "n_kpts 3"]
) -> Float[ndarray, "3"]:
    """Estimate MANO global orientation from triangulated joints using orthogonal Procrustes.

    The returned rotation is a Rodrigues axis-angle vector compatible with the ``MANOLayerNP``
    global orientation parameter. Following [Pavlakos et al. 2019], we align the template MANO
    joints with the observed joints by matching the unit vectors from the wrist to the four
    non-thumb MCP joints together with the hand normal. The optimal rotation in :math:`SO(3)` is
    found via the SVD-based closed form solution to the orthogonal Procrustes problem.
    """

    root_idx: int = 0
    finger_mcp_indices: tuple[int, int, int, int] = (5, 9, 13, 17)
    eps: float = 1e-8

    def compute_direction_matrix(xyz: Float[ndarray, "n_kpts 3"]) -> Float[ndarray, "3 5"]:
        """Return unit wrist directions to the MCP joints plus a palm normal."""

        basis: Float[ndarray, "3 5"] = np.full((3, 5), np.nan, dtype=np.float64)
        if xyz.shape[0] <= root_idx:
            return basis

        # Column 0 is treated as the wrist origin for all direction vectors.
        root: Float[ndarray, "3"] = xyz[root_idx].astype(np.float64, copy=False)
        if not np.all(np.isfinite(root)):
            return basis

        # Columns 0–3 store unit vectors from the wrist to each finger's MCP joint.
        for column_idx, joint_idx in enumerate(finger_mcp_indices):
            if joint_idx >= xyz.shape[0]:
                continue
            joint: Float[ndarray, "3"] = xyz[joint_idx].astype(np.float64, copy=False)
            if not np.all(np.isfinite(joint)):
                continue
            vector: Float[ndarray, "3"] = joint - root
            length: float = float(np.linalg.norm(vector))
            if length <= eps:
                continue
            basis[:, column_idx] = vector / length

        # Collect the finite MCP directions so we can safely form cross products.
        valid_vectors: list[Float[ndarray, "3"]] = [
            basis[:, col].copy() for col in range(4) if np.all(np.isfinite(basis[:, col]))
        ]

        # Column 4 captures an estimated palm normal, falling back to any valid pair if needed.
        normal: Float[ndarray, "3"] | None = None
        if np.all(np.isfinite(basis[:, 0])) and np.all(np.isfinite(basis[:, 3])):
            candidate: Float[ndarray, "3"] = np.cross(basis[:, 0], basis[:, 3])
            candidate_norm: float = float(np.linalg.norm(candidate))
            if candidate_norm > eps:
                normal = candidate / candidate_norm
        if normal is None:
            for i in range(len(valid_vectors)):
                for j in range(i + 1, len(valid_vectors)):
                    candidate = np.cross(valid_vectors[i], valid_vectors[j])
                    candidate_norm = float(np.linalg.norm(candidate))
                    if candidate_norm > eps:
                        normal = candidate / candidate_norm
                        break
                if normal is not None:
                    break

        if normal is not None:
            basis[:, 4] = normal
        return basis

    mano_xyz_f64: Float[ndarray, "n_kpts 3"] = xyz_mano.astype(np.float64, copy=False)
    triangulated_xyz_f64: Float[ndarray, "n_kpts 3"] = xyz_triangulated.astype(np.float64, copy=False)

    # Build orthonormal wrist-to-finger bases for the canonical MANO pose and the observations.
    mano_basis: Float[ndarray, "3 5"] = compute_direction_matrix(mano_xyz_f64)
    triangulated_basis: Float[ndarray, "3 5"] = compute_direction_matrix(triangulated_xyz_f64)

    # Keep only shared, finite basis vectors so the Procrustes fit uses comparable directions.
    valid_mask: ndarray = (~np.isnan(mano_basis).any(axis=0)) & (~np.isnan(triangulated_basis).any(axis=0))
    valid_count: int = int(np.count_nonzero(valid_mask))
    if valid_count < 2:
        # Fewer than two directions leave the wrist frame under-constrained; return identity.
        zero_rotation: Float32[ndarray, "3"] = np.zeros((3,), dtype=np.float32)
        return zero_rotation

    mano_basis_valid: Float[ndarray, "3 valid"] = mano_basis[:, valid_mask]
    triangulated_basis_valid: Float[ndarray, "3 valid"] = triangulated_basis[:, valid_mask]
    # Compute the best-fit rotation between the two bases via a 3x3 Procrustes solve.
    covariance: Float[ndarray, "3 3"] = triangulated_basis_valid @ mano_basis_valid.T
    svd_result: tuple[
        Float[ndarray, "3 3"],
        Float[ndarray, "3"],
        Float[ndarray, "3 3"],
    ] = np.linalg.svd(covariance, full_matrices=True)
    u_matrix: Float[ndarray, "3 3"] = svd_result[0]
    _singular_values: Float[ndarray, "3"] = svd_result[1]
    vt_matrix: Float[ndarray, "3 3"] = svd_result[2]

    # Enforce a proper rotation (determinant +1) before extracting Rodrigues parameters.
    uv_t: Float[ndarray, "3 3"] = u_matrix @ vt_matrix
    det_uv_t: float = float(np.linalg.det(uv_t))
    corrected_diag: Float[ndarray, "3 3"] = np.eye(3, dtype=np.float64)
    corrected_diag[2, 2] = 1.0 if det_uv_t >= 0.0 else -1.0

    rotation_matrix: Float[ndarray, "3 3"] = u_matrix @ corrected_diag @ vt_matrix

    # Rodrigues gives the axis-angle vector expected by MANO from the corrected rotation matrix.
    rotvec_matrix: Float[ndarray, "3 1"] = cv2.Rodrigues(rotation_matrix.astype(np.float64, copy=False))[0]
    rotation_vec: Float32[ndarray, "3"] = rotvec_matrix.reshape(-1).astype(np.float32, copy=False)
    return rotation_vec


@dataclass(slots=True)
class TriangulationResult:
    """Triangulated keypoints inferred from multi-view observations."""

    xyz: Float[ndarray, "n_kpts=133 3"]
    """COCO-133 keypoint positions in world coordinates."""
    confidences: Float[ndarray, "n_kpts=133"]
    """Confidence score per COCO-133 keypoint derived from triangulation view counts."""


@dataclass(slots=True)
class ManoOptimizationResult:
    """Optimised MANO pose/shape parameters to recreate vertices on demand."""

    so3: Float32[ndarray, "n_frames 16 3"]
    """Axis-angle rotations for the global orientation and 15 joints."""
    translation: Float32[ndarray, "n_frames 3"]
    """Per-frame MANO wrist translation in world coordinates."""
    betas: Float32[ndarray, "10"]
    """Shape coefficients describing the personalised MANO mesh."""
    faces: Int[ndarray, "n_faces=1538 3"]
    """Triangle indices describing the MANO mesh topology."""


@dataclass(slots=True)
class HandCalibrationResult:
    """Outputs from a single calibration pass for downstream consumers."""

    pinhole_param_list: list[PinholeParameters]
    """Per-camera pinhole parameters aligned with the exo rig ordering."""
    xyz: Float[ndarray, "n_kpts=133 3"]
    """Triangulated 3D COCO-133 keypoints at the calibrated timestamp."""
    confidences: Float[ndarray, "n_kpts=133"]
    """Confidence score per COCO-133 keypoint derived from multi-view triangulation."""
    mano: ManoOptimizationResult | None = None
    """MANO pose/shape parameters (and topology) when optimisation runs; ``None`` otherwise."""


@dataclass(slots=True)
class DetectionResults:
    """Convenience container bundling multi-view detections for triangulation."""

    uvc_coco_batch: Float[ndarray, "n_views n_kpts=133 3"]
    """Stack of per-camera COCO-133 detections with confidence columns."""
    right_hand_kpts: KeypointResults | None
    """Most confident right-hand keypoint detection, if present."""
    left_hand_kpts: KeypointResults | None
    """Most confident left-hand keypoint detection, if present."""


@dataclass
class HandCalibratorConfig:
    """Configuration parameters steering multi-view MANO optimisation (pipeline step 3)."""

    hand_side: Literal["left", "right"] = "right"
    """Which hand to optimize with MANO—either "left" or "right"."""
    detection_confidence: float = 0.5
    """Minimum detector confidence required to keep a hand bounding box."""
    ts_nano: int | None = None
    """Optional absolute timestamp in nanoseconds to sample; defaults to first frame when ``None``."""
    mano_optim_iters: int = 60
    """Number of Levenberg-Marquardt iterations for MANO pose/shape fitting."""
    n_frame_optim: int = 1
    """Number of consecutive frames to jointly optimize."""
    verbose: bool = True
    """Whether to log additional intermediate results for debugging purposes."""


class HandCalibrator:
    def __init__(
        self,
        hand_detector: HandDetector,
        hand_keypoint_detector: WilorHandKeypointDetector,
        config: HandCalibratorConfig,
    ) -> None:
        self.hand_detector: HandDetector = hand_detector
        self.hand_keypoint_detector: WilorHandKeypointDetector = hand_keypoint_detector
        self.config: HandCalibratorConfig = config

    def __call__(
        self,
        *,
        exo_cam_list: list[PinholeParameters],
        rgb_ts_batch: UInt8[ndarray, "n_frames n_views H W 3"],
        parent_log_path: Path,
    ) -> HandCalibrationResult:
        calibration_root: Path = parent_log_path / "hand_calibration"

        rgb_list: UInt8[ndarray, "n_views H W 3"] = rgb_ts_batch[0]

        parsed_detections: DetectionResults = self._detect_keypoints(
            rgb_list=rgb_list,
            pinhole_param_list=exo_cam_list,
            parent_log_path=parent_log_path,
        )

        triangulation_result: TriangulationResult = self._triangulate_keypoints(
            parsed_detections.uvc_coco_batch,
            exo_cam_list,
            calibration_root=calibration_root,
        )
        xyz: Float[ndarray, "n_kpts=133 3"] = triangulation_result.xyz
        conf_values: Float[ndarray, "n_kpts=133"] = triangulation_result.confidences

        mano_result: ManoOptimizationResult = self._optimize_mano(
            xyz=xyz,
            confidences=conf_values,
            parsed_detections=parsed_detections,
            pinhole_param_list=exo_cam_list,
            calibration_root=calibration_root,
        )

        return HandCalibrationResult(
            pinhole_param_list=exo_cam_list,
            xyz=xyz,
            confidences=conf_values,
            mano=mano_result,
        )

    def _detect_keypoints(
        self,
        *,
        rgb_list: UInt8[ndarray, "n_views H W 3"],
        pinhole_param_list: list[PinholeParameters],
        parent_log_path: Path,
    ) -> DetectionResults:
        uvc_coco_list: list[Float[ndarray, "coco_kpts=133 3"]] = []
        right_hand_kpts: KeypointResults | None = None
        left_hand_kpts: KeypointResults | None = None

        for camera_idx, (rgb_hw3, pinhole) in enumerate(zip(rgb_list, pinhole_param_list, strict=True)):
            det_result: DetectionResult = self.hand_detector(
                rgb_hw3=rgb_hw3,
                hand_conf=self.config.detection_confidence,
            )
            camera_name: str = getattr(pinhole, "name", f"camera_{camera_idx}")
            hand_path_root: Path = parent_log_path / "exo" / camera_name / "pinhole"
            uvc_coco: Float[ndarray, "coco_kpts=133 3"] = np.full((133, 3), np.nan, dtype=np.float32)

            for hand_label, xyxy in ("left", det_result.left_xyxy), ("right", det_result.right_xyxy):
                if xyxy is None:
                    continue

                hand_log_path: Path = hand_path_root / "video" / hand_label
                kpts_results: KeypointResults = self.hand_keypoint_detector(
                    rgb_hw3=rgb_hw3,
                    xyxy=xyxy,
                    handedness=hand_label,
                )
                uv: Float[ndarray, "n_frames=1 mp_kpts=21 2"] = kpts_results.keypoints_2d
                conf: Float[ndarray, "n_frames=1 mp_kpts=21"] = kpts_results.scores
                conf_values: Float[ndarray, "mp_kpts=21"] = conf[0].astype(np.float32)
                mean_conf: float = float(np.nanmean(conf_values))
                uv_filtered: Float[ndarray, "mp_kpts=21 2"] = uv[0].copy()
                if mean_conf < 0.4:
                    uv_filtered[:] = np.nan
                    conf_values[:] = 0.0

                conf_colors: UInt8[ndarray, "n_frames=1 mp_kpts=21 3"] = confidence_scores_to_rgb(
                    confidence_scores=conf_values[np.newaxis, :, np.newaxis]
                )

                fill_idx = RIGHT_HAND_IDX if hand_label == "right" else LEFT_HAND_IDX
                uvc_coco[fill_idx, :2] = uv_filtered
                uvc_coco[fill_idx, 2] = conf_values
                if hand_label == "right" and right_hand_kpts is None:
                    right_hand_kpts = kpts_results
                if hand_label == "left" and left_hand_kpts is None:
                    left_hand_kpts = kpts_results

                if self.config.verbose:
                    rr.log(
                        f"{hand_log_path}/bbox",
                        rr.Boxes2D(
                            array=xyxy,
                            array_format=rr.Box2DFormat.XYXY,
                            class_ids=0 if hand_label == "left" else 1,
                            show_labels=True,
                        ),
                    )
                    rr.log(
                        f"{hand_log_path}/keypoints",
                        Points2DWithConfidence(
                            positions=uv_filtered,
                            confidences=conf_values,
                            class_ids=0 if hand_label == "left" else 1,
                            keypoint_ids=MEDIAPIPE_IDS,
                            show_labels=False,
                            colors=conf_colors[0],
                        ),
                    )

            uvc_coco_list.append(uvc_coco)

        uvc_coco_batch: Float[ndarray, "n_views coco_kpts=133 3"] = np.stack(uvc_coco_list)
        return DetectionResults(
            uvc_coco_batch=uvc_coco_batch,
            right_hand_kpts=right_hand_kpts,
            left_hand_kpts=left_hand_kpts,
        )

    def _triangulate_keypoints(
        self,
        uvc_coco_batch: Float[ndarray, "n_views coco_kpts=133 3"],
        pinhole_param_list: list[PinholeParameters],
        *,
        calibration_root: Path,
    ) -> TriangulationResult:
        uvc_triangulate_batch: Float[ndarray, "n_views coco_kpts=133 3"] = np.nan_to_num(
            uvc_coco_batch.copy(),
            nan=0.0,
        )
        Pall_exo: Float[ndarray, "n_views 3 4"] = np.stack(
            [pinhole.projection_matrix for pinhole in pinhole_param_list]
        ).astype(np.float32)
        xyzc: Float[ndarray, "coco_kpts=133 4"] = batch_triangulate(
            uvc_triangulate_batch,
            Pall_exo,
            min_views=2,
        )
        xyz: Float[ndarray, "coco_kpts=133 3"] = xyzc[:, :3]
        conf_values: Float[ndarray, "coco_kpts=133"] = xyzc[:, 3]
        xyz_for_logging: Float[ndarray, "coco_kpts=133 3"] = np.where(
            conf_values[:, np.newaxis] > 0,
            xyz,
            np.nan,
        )
        conf_colors: UInt8[ndarray, "1 coco_kpts=133 3"] = confidence_scores_to_rgb(
            conf_values[:, np.newaxis][np.newaxis, ...]
        )

        rr.log(
            f"{calibration_root}/wb_keypoints",
            Points3DWithConfidence(
                positions=xyz_for_logging,
                confidences=conf_values,
                class_ids=2,
                keypoint_ids=COCO_133_IDS,
                show_labels=False,
                colors=conf_colors[0],
            ),
        )
        triangulation_result = TriangulationResult(
            xyz=xyz,
            confidences=conf_values,
        )
        return triangulation_result

    def _optimize_mano(
        self,
        *,
        xyz: Float[ndarray, "coco_kpts=133 3"],
        confidences: Float[ndarray, "coco_kpts=133"],
        parsed_detections: DetectionResults,
        pinhole_param_list: list[PinholeParameters],
        calibration_root: Path,
    ) -> ManoOptimizationResult:
        hand_side: Literal["left", "right"] = self.config.hand_side
        uv_exo_stack: Float[ndarray, "n_frames=1 n_views n_kpts=133 2"] = parsed_detections.uvc_coco_batch[
            np.newaxis, :, :, 0:2
        ]

        Pall_exo: Float[ndarray, "n_views 3 4"] = np.stack(
            [pinhole.projection_matrix for pinhole in pinhole_param_list]
        ).astype(np.float32)

        kpts_results_selected: KeypointResults | None = (
            parsed_detections.right_hand_kpts if hand_side == "right" else parsed_detections.left_hand_kpts
        )
        if kpts_results_selected is None:
            raise ValueError(f"No keypoint detections available for hand '{hand_side}'")

        if kpts_results_selected.global_orient is None or kpts_results_selected.hand_pose is None:
            so3_init: Float[ndarray, "b 48"] = np.zeros((self.config.n_frame_optim, 48), dtype=np.float32)
        else:
            global_orient: Float[ndarray, "b 1 3"] = kpts_results_selected.global_orient.astype(np.float32)
            hand_pose: Float[ndarray, "b 15 3"] = kpts_results_selected.hand_pose.astype(np.float32)
            so3_concat: Float[ndarray, "b 16 3"] = np.concatenate([global_orient, hand_pose], axis=1)
            so3_init = so3_concat.reshape(so3_concat.shape[0], -1)
            if so3_init.shape[0] != self.config.n_frame_optim:
                if so3_init.shape[0] == 0:
                    so3_init = np.zeros((self.config.n_frame_optim, 48), dtype=np.float32)
                else:
                    so3_init = np.broadcast_to(so3_init[0:1], (self.config.n_frame_optim, so3_init.shape[1])).astype(
                        np.float32,
                        copy=True,
                    )
            else:
                so3_init = so3_init.astype(np.float32, copy=False)

        hand_indices = RIGHT_HAND_IDX if hand_side == "right" else LEFT_HAND_IDX
        hand_xyz: Float[ndarray, "n_hand 3"] = xyz[hand_indices]
        hand_conf: Float[ndarray, "n_hand"] = confidences[hand_indices]
        valid_hand_mask: Bool[ndarray, "..."] = hand_conf > 0.0  # type: ignore Pyrefly bug
        if np.any(valid_hand_mask):
            trans_guess: Float[ndarray, "3"] = hand_xyz[valid_hand_mask].mean(axis=0).astype(np.float32)
        else:
            trans_guess = np.zeros((3,), dtype=np.float32)
        trans_init: Float[ndarray, "b 3"] = np.broadcast_to(trans_guess, (self.config.n_frame_optim, 3)).astype(
            np.float32, copy=True
        )

        if kpts_results_selected.betas is not None and kpts_results_selected.betas.size > 0:
            betas_array = np.asarray(kpts_results_selected.betas, dtype=np.float32)
            if betas_array.ndim == 1:
                beta_init: Float[ndarray, "10"] = betas_array
            else:
                beta_init = betas_array[0]
        else:
            beta_init = np.zeros((10,), dtype=np.float32)

        mano_init_layer = MANOLayerNP(side=hand_side, betas=beta_init.astype(np.float32, copy=False))

        so3_init_naive: Float32[ndarray, "b 48"] = so3_init.astype(np.float32, copy=True)
        trans_init_f32: Float32[ndarray, "b 3"] = trans_init.astype(np.float32, copy=False)
        mano_naive_results: tuple[
            Float32[ndarray, "n_frames n_verts=778 3"],
            Float32[ndarray, "n_frames mp_kpts=21 3"],
        ] = mano_init_layer(so3_init_naive, trans_init_f32)
        verts_naive: Float32[ndarray, "n_frames n_verts=778 3"] = mano_naive_results[0]
        joints_naive: Float32[ndarray, "n_frames mp_kpts=21 3"] = mano_naive_results[1]

        hand_xyz_masked: Float32[ndarray, "n_hand 3"] = hand_xyz.astype(np.float32, copy=True)
        hand_xyz_masked[~valid_hand_mask] = np.nan
        rotation_vec: Float32[ndarray, "3"] = align_rotation(joints_naive[0], hand_xyz_masked)
        rotation_matrix_debug: Float[ndarray, "3 3"] = cv2.Rodrigues(rotation_vec)[0]
        rotated_joints_debug: Float32[ndarray, "mp_kpts=21 3"] = (
            (joints_naive[0] - joints_naive[0, 0]) @ rotation_matrix_debug.T
        ) + joints_naive[0, 0]

        valid_indices_debug: np.ndarray = np.isfinite(hand_xyz_masked).all(axis=1)
        if np.any(valid_indices_debug):
            translation_offset: Float32[ndarray, "3"] = (
                np.nanmean(hand_xyz_masked[valid_indices_debug] - rotated_joints_debug[valid_indices_debug], axis=0)
            ).astype(np.float32)
        else:
            translation_offset = np.zeros((3,), dtype=np.float32)

        rotated_joints_debug = rotated_joints_debug + translation_offset[np.newaxis, :]

        so3_init_aligned: Float32[ndarray, "b 48"] = so3_init_naive.copy()
        for frame_idx in range(so3_init_aligned.shape[0]):
            naive_rotvec: Float32[ndarray, "3"] = so3_init_naive[frame_idx, 0:3]
            naive_matrix: Float[ndarray, "3 3"] = cv2.Rodrigues(naive_rotvec)[0]
            composed_matrix: Float[ndarray, "3 3"] = rotation_matrix_debug @ naive_matrix
            composed_rotvec: Float[ndarray, "3 1"] = cv2.Rodrigues(composed_matrix)[0]
            so3_init_aligned[frame_idx, 0:3] = composed_rotvec.reshape(3).astype(np.float32)

        mano_aligned_pre_results: tuple[
            Float32[ndarray, "n_frames n_verts=778 3"],
            Float32[ndarray, "n_frames mp_kpts=21 3"],
        ] = mano_init_layer(so3_init_aligned, trans_init_f32)

        verts_aligned_pre: Float32[ndarray, "n_frames n_verts=778 3"] = mano_aligned_pre_results[0]
        joints_aligned_pre: Float32[ndarray, "n_frames mp_kpts=21 3"] = mano_aligned_pre_results[1]

        root_delta: Float32[ndarray, "3"] = (joints_naive[0, 0] - joints_aligned_pre[0, 0]).astype(np.float32)
        trans_init_aligned: Float32[ndarray, "b 3"] = (
            trans_init_f32 + root_delta[np.newaxis, :] + translation_offset[np.newaxis, :]
        )

        mano_aligned_init_results: tuple[
            Float32[ndarray, "n_frames n_verts=778 3"],
            Float32[ndarray, "n_frames mp_kpts=21 3"],
        ] = mano_init_layer(so3_init_aligned, trans_init_aligned)
        verts_aligned_init: Float32[ndarray, "n_frames n_verts=778 3"] = mano_aligned_init_results[0]
        joints_aligned_init: Float32[ndarray, "n_frames mp_kpts=21 3"] = mano_aligned_init_results[1]

        so3_init = so3_init_aligned
        trans_init = trans_init_aligned

        optim_shape_cfg = PoseShapeOptimConfig(
            Pall=Pall_exo,
            hand_side=hand_side,
            loss_weights=LossWeights(
                keypoint_2d=1.0,
                depth=0.0,
                temp=0.0,
                pose_reg=1.0,
            ),
            n_frames_optim=self.config.n_frame_optim,
            n_optim_iters=self.config.mano_optim_iters,
        )
        optimizer_shape = SingleHandShapeOptim(config=optim_shape_cfg)

        optim_shape_input: OptimShapeInput = OptimShapeInput(
            uv_pred=uv_exo_stack,
            beta_init=beta_init,
            so3_init=so3_init,
            trans_init=trans_init,
        )
        optim_shape_tuple: tuple[OptimShapeResult, LevenbergMarquardtState] = optimizer_shape(optim_shape_input)
        optim_shape_result: OptimShapeResult = optim_shape_tuple[0]

        beta_optim: Float32[ndarray, "10"] = optim_shape_result.beta_optim.astype(np.float32, copy=False)

        mano_optim_layer = MANOLayerNP(side=hand_side, betas=beta_optim)

        so3_optim_flat: Float32[ndarray, "n_frames 48"] = optim_shape_result.so3_optim.astype(np.float32, copy=False)
        trans_optim: Float32[ndarray, "n_frames 3"] = optim_shape_result.trans_optim.astype(np.float32, copy=False)

        mano_optim_results: tuple[
            Float32[ndarray, "n_frames n_verts=778 3"],
            Float32[ndarray, "n_frames mp_kpts=21 3"],
        ] = mano_optim_layer(
            so3_optim_flat,
            trans_optim,
        )
        verts_optim: Float32[ndarray, "n_frames n_verts=778 3"] = mano_optim_results[0]
        joints_optim: Float32[ndarray, "n_frames mp_kpts=21 3"] = mano_optim_results[1]

        so3_optim: Float32[ndarray, "n_frames 16 3"] = so3_optim_flat.reshape(so3_optim_flat.shape[0], 16, 3)

        faces_np: Int[ndarray, "n_faces=1538 3"] = mano_optim_layer.f.astype(np.int32)
        normals_naive: Float32[ndarray, "n_frames n_verts=778 3"] = compute_vertex_normals_batch(
            verts_naive[0:1], faces_np
        )
        normals_aligned: Float32[ndarray, "n_frames n_verts=778 3"] = compute_vertex_normals_batch(
            verts_aligned_init[0:1], faces_np
        )
        normals_optim: Float32[ndarray, "n_frames n_verts=778 3"] = compute_vertex_normals_batch(
            verts_optim[0:1], faces_np
        )

        mano_mesh_path: Path = calibration_root / f"{hand_side}_mano_mesh"
        mano_joint_path: Path = calibration_root / f"{hand_side}_mano_xyz"

        verts_naive_frame: Float32[ndarray, "n_verts=778 3"] = verts_naive[0]
        joints_naive_frame: Float32[ndarray, "mp_kpts=21 3"] = joints_naive[0]
        verts_aligned_frame: Float32[ndarray, "n_verts=778 3"] = verts_aligned_init[0]
        joints_aligned_frame: Float32[ndarray, "mp_kpts=21 3"] = joints_aligned_init[0]
        verts_aligned_optim: Float32[ndarray, "n_verts=778 3"] = verts_optim[0]
        joints_aligned_optim: Float32[ndarray, "mp_kpts=21 3"] = joints_optim[0]

        class_id: int = 1 if hand_side == "right" else 0
        class_ids: Int[ndarray, "mp_kpts=21"] = np.full((joints_aligned_optim.shape[0],), class_id, dtype=np.int32)
        keypoint_ids: Int[ndarray, "mp_kpts=21"] = np.asarray(MEDIAPIPE_IDS, dtype=np.int32)

        if self.config.verbose:
            rr.log(
                f"{mano_mesh_path}_init",
                rr.Mesh3D(
                    vertex_positions=verts_naive_frame,
                    triangle_indices=faces_np,
                    vertex_normals=normals_naive[0],
                    albedo_factor=(255, 64, 0, 255),
                ),
            )
            rr.log(
                f"{mano_mesh_path}_aligned",
                rr.Mesh3D(
                    vertex_positions=verts_aligned_frame,
                    triangle_indices=faces_np,
                    vertex_normals=normals_aligned[0],
                    albedo_factor=(0, 255, 0, 255),
                ),
            )
        rr.log(
            f"{mano_mesh_path}_optim",
            rr.Mesh3D(
                vertex_positions=verts_aligned_optim,
                triangle_indices=faces_np,
                vertex_normals=normals_optim[0],
                albedo_factor=(0, 0, 255, 255),
            ),
        )
        rr.log(
            f"{mano_joint_path}",
            rr.Points3D(
                joints_aligned_optim,
                class_ids=class_ids,
                keypoint_ids=keypoint_ids,
                show_labels=False,
            ),
        )
        if self.config.verbose:
            rr.log(
                f"{mano_joint_path}_init",
                rr.Points3D(
                    joints_naive_frame,
                    class_ids=class_ids,
                    keypoint_ids=keypoint_ids,
                    show_labels=False,
                ),
            )
            rr.log(
                f"{mano_joint_path}_aligned",
                rr.Points3D(
                    joints_aligned_frame,
                    class_ids=class_ids,
                    keypoint_ids=keypoint_ids,
                    show_labels=False,
                ),
            )

        mano_result = ManoOptimizationResult(
            so3=so3_optim,
            translation=trans_optim,
            betas=beta_optim,
            faces=faces_np,
        )
        return mano_result
