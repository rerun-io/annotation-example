from dataclasses import dataclass
from pathlib import Path

import numpy as np
import rerun as rr
from jaxtyping import Float, Float32, UInt8
from numpy import ndarray
from simplecv.camera_parameters import PinholeParameters
from wilor_nano.hand_detection import DetectionResult, HandDetector
from wilor_nano.hand_keypoints import WilorHandKeypointDetector


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
    verbose: bool = False
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

    def __call__(
        self,
        *,
        rgb_batch: UInt8[ndarray, "n_views H W 3"],
        pinhole_param_list: list[PinholeParameters],
        recording: rr.RecordingStream | None,
    ) -> None:
        """Execute the multi-view hand tracking loop (implementation pending)."""
        # detect hands in each view
        ...

    def _detect_hands(
        self,
        *,
        rgb_batch: UInt8[ndarray, "n_views H W 3"],
        pinhole_param_list: list[PinholeParameters],
    ) -> MVDetectionResults:
        left_xyxy_list: list[Float[ndarray, "1 4"]] = []
        right_xyxy_list: list[Float[ndarray, "1 4"]] = []

        for rgb_hw3 in rgb_batch:
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

        left_xyxy_batch: Float[ndarray, "n_views 1 4"] = np.stack(left_xyxy_list, axis=0)
        right_xyxy_batch: Float[ndarray, "n_views 1 4"] = np.stack(right_xyxy_list, axis=0)

        return MVDetectionResults(
            left_xyxy_batch=left_xyxy_batch,
            right_xyxy_batch=right_xyxy_batch,
        )
