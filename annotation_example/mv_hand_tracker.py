from dataclasses import dataclass
from pathlib import Path

import numpy as np
import rerun as rr
from jaxtyping import Float, Float32, UInt8
from numpy import ndarray
from simplecv.camera_parameters import PinholeParameters
from simplecv.data.skeleton.coco_133 import LEFT_HAND_IDX, RIGHT_HAND_IDX
from simplecv.data.skeleton.mediapipe import MEDIAPIPE_IDS
from simplecv.rerun_log_utils import Points2DWithConfidence, confidence_scores_to_rgb
from wilor_nano.hand_detection import DetectionResult, HandDetector
from wilor_nano.hand_keypoints import KeypointResults, WilorHandKeypointDetector


@dataclass
class MVDetectionResults:
    """Container for multi-view detections"""

    left_xyxy_batch: Float[ndarray, "n_views 1 4"]
    """Batch of left-hand bounding boxes in (x_min, y_min, x_max, y_max) format."""
    right_xyxy_batch: Float[ndarray, "n_views 1 4"]
    """Batch of right-hand bounding boxes in (x_min, y_min, x_max, y_max) format."""


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
        self._latest_uvc_coco: Float[ndarray, "n_views n_kpts=133 3"] | None = None

    def __call__(
        self,
        *,
        rgb_batch: UInt8[ndarray, "n_views H W 3"],
        pinhole_param_list: list[PinholeParameters],
        recording: rr.RecordingStream | None = None,
    ) -> None:
        """Run hand detection followed by per-view keypoint refinement."""

        detection_results: MVDetectionResults = self._detect_hands(
            rgb_batch=rgb_batch,
            pinhole_param_list=pinhole_param_list,
            recording=recording,
        )

        uvc_coco_batch: Float[ndarray, "n_views n_kpts=133 3"] = self._detect_keypoints(
            rgb_batch=rgb_batch,
            detections=detection_results,
            pinhole_param_list=pinhole_param_list,
            recording=recording,
        )

        self._latest_uvc_coco = uvc_coco_batch

    def _detect_hands(
        self,
        *,
        rgb_batch: UInt8[ndarray, "n_views H W 3"],
        pinhole_param_list: list[PinholeParameters],
        recording: rr.RecordingStream | None = None,
    ) -> MVDetectionResults:
        left_xyxy_list: list[Float[ndarray, "1 4"]] = []
        right_xyxy_list: list[Float[ndarray, "1 4"]] = []

        for pinhole_param, rgb_hw3 in zip(pinhole_param_list, rgb_batch, strict=True):
            det_result: DetectionResult = self.hand_detector(
                rgb_hw3=rgb_hw3,
                hand_conf=self.config.detection_confidence,
            )
            # fmt: off
            left_xyxy: Float[ndarray, "1 4"] = (
                det_result.left_xyxy
                if det_result.left_xyxy is not None
                else np.full((1, 4), np.nan, dtype=np.float32)
            )
            right_xyxy: Float[ndarray, "1 4"] = (
                det_result.right_xyxy
                if det_result.right_xyxy is not None
                else np.full((1, 4), np.nan, dtype=np.float32)
            )
            # fmt: on
            left_xyxy_list.append(left_xyxy)
            right_xyxy_list.append(right_xyxy)
            if self.config.verbose:
                cam_log_path: Path = self.parent_log_path / "exo" / pinhole_param.name
                pinhole_log_path: Path = cam_log_path / "pinhole"
                rr.log(
                    f"{pinhole_log_path}/left_bbox",
                    rr.Boxes2D(array=left_xyxy, array_format=rr.Box2DFormat.XYXY, class_ids=0),
                    recording=recording,
                )
                rr.log(
                    f"{pinhole_log_path}/right_bbox",
                    rr.Boxes2D(array=right_xyxy, array_format=rr.Box2DFormat.XYXY, class_ids=1),
                    recording=recording,
                )

        left_xyxy_batch: Float[ndarray, "n_views 1 4"] = np.stack(left_xyxy_list, axis=0)
        right_xyxy_batch: Float[ndarray, "n_views 1 4"] = np.stack(right_xyxy_list, axis=0)

        return MVDetectionResults(
            left_xyxy_batch=left_xyxy_batch,
            right_xyxy_batch=right_xyxy_batch,
        )

    def _detect_keypoints(
        self,
        *,
        rgb_batch: UInt8[ndarray, "n_views H W 3"],
        detections: MVDetectionResults,
        pinhole_param_list: list[PinholeParameters],
        recording: rr.RecordingStream | None = None,
    ) -> Float[ndarray, "n_views n_kpts=133 3"]:
        """Populate COCO-133 keypoints per view for both hands."""

        n_views: int = int(rgb_batch.shape[0])
        uvc_batch: Float[ndarray, "n_views n_kpts=133 3"] = np.full((n_views, 133, 3), np.nan, dtype=np.float32)

        for view_idx, (rgb_hw3, left_xyxy, right_xyxy, pinhole_param) in enumerate(
            zip(
                rgb_batch,
                detections.left_xyxy_batch,
                detections.right_xyxy_batch,
                pinhole_param_list,
                strict=True,
            )
        ):
            for hand_label, bbox_xyxy, fill_indices in (
                ("left", left_xyxy, LEFT_HAND_IDX),
                ("right", right_xyxy, RIGHT_HAND_IDX),
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

                uvc_batch[view_idx, fill_indices, :2] = uv_21x2
                uvc_batch[view_idx, fill_indices, 2] = conf_values

                if self.config.verbose:
                    hand_log_path: Path = (
                        self.parent_log_path / "exo" / pinhole_param.name / "pinhole" / "video" / hand_label
                    )
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

        return uvc_batch

    def _triangulate_keypoints(
        self,
        *,
        uvc_batch: Float[ndarray, "n_views n_kpts=133 3"],
        pinhole_param_list: list[PinholeParameters],
        recording: rr.RecordingStream | None = None,
    ) -> Float[ndarray, "n_kpts=133 3"]:
        """
        Triangulate the multi-view keypoints into 3D space. The uvc_batch can contain NaNs for views where no hand was detected.
        The output is the triangulated keypoints in 3D space, with NaNs for keypoints that could not be triangulated.
        """
        raise NotImplementedError("Triangulation not yet implemented")

    def _fit_mano_model(
        self,
        *,
        keypoints_3d: Float[ndarray, "n_kpts=133 3"],
    ) -> None:
        """Fit the MANO model to the triangulated 3D keypoints. The keypoints_3d can contain NaNs for keypoints that could not be triangulated."""
        raise NotImplementedError("MANO fitting not yet implemented")
