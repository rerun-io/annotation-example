from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import numpy as np
import rerun as rr
from jaxtyping import Float, Float32, UInt8
from numpy import ndarray
from simplecv.camera_parameters import PinholeParameters
from simplecv.data.skeleton.mediapipe import MEDIAPIPE_IDS
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
        parent_log_path: Path,
    ) -> None:
        """Persist the tracker configuration for later inference runs."""
        self.config: MultiViewHandTrackerConfig = config
        self.hand_detector: HandDetector = hand_detector
        self.hand_keypoint_detector: WilorHandKeypointDetector = hand_keypoint_detector
        self.betas: Float32[ndarray, "10"] = betas
        self.parent_log_path: Path = parent_log_path

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

            xyzc: Float32[ndarray, "mp_kpts=21 4"] | None = self._triangulate_keypoints(
                hand_uvc=uvc_batch,
                pinhole_param_list=pinhole_param_list,
                hand_label=hand_label,
                recording=recording,
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
        xyz: Float[ndarray, "mp_kpts=21 3"],
    ) -> ManoResults:
        """Fit the MANO model to the triangulated 3D keypoints. The keypoints_3d can contain NaNs for keypoints that could not be triangulated."""
        raise NotImplementedError("MANO fitting not yet implemented")
