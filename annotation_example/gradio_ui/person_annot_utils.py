import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, TypedDict

import numpy as np
import rerun as rr
from jaxtyping import Bool, Float, UInt8
from numpy import ndarray
from serde import field as sfield
from serde import serde
from simplecv.camera_parameters import PinholeParameters

SELECTED_COLOR = (255, 0, 0)  # RED
SELECTED_XYXY_COLOR = (144, 238, 144)


def get_recording(recording_id: uuid.UUID) -> rr.RecordingStream:
    return rr.RecordingStream(application_id="rerun_example_gradio", recording_id=recording_id)


class RerunLogPaths(TypedDict):
    timeline_name: str
    parent_log_path: Path
    camera_log_path: Path
    pinhole_log_path: Path
    original_image_path: Path
    annotated_image_path: Path


@dataclass
class SAM2KeypointContainer:
    """Container for include and exclude keypoints"""

    include_points: np.ndarray  # shape (n,2)
    exclude_points: np.ndarray  # shape (m,2)

    @classmethod
    def empty(cls) -> "SAM2KeypointContainer":
        """Create an empty keypoints container"""
        return cls(include_points=np.zeros((0, 2), dtype=float), exclude_points=np.zeros((0, 2), dtype=float))

    def add_point(self, point: tuple[float, float], label: Literal["include", "exclude"]) -> None:
        """Add a point with the specified label"""
        point_array = np.array([point], dtype=float)
        if label == "include":
            self.include_points = (
                np.vstack([self.include_points, point_array]) if self.include_points.shape[0] > 0 else point_array
            )
        else:
            self.exclude_points = (
                np.vstack([self.exclude_points, point_array]) if self.exclude_points.shape[0] > 0 else point_array
            )

    def clear(self) -> None:
        """Clear all points"""
        self.include_points = np.zeros((0, 2), dtype=float)
        self.exclude_points = np.zeros((0, 2), dtype=float)


@dataclass
class XYXYContainer:
    """Container for include and exclude keypoints"""

    top_left: np.ndarray  # shape (n,2)
    bottom_right: np.ndarray  # shape (m,2)
    bbox: np.ndarray | None = None  # shape (n,4) if bbox is logged

    @classmethod
    def empty(cls) -> "XYXYContainer":
        """Create an empty keypoints container"""
        return cls(top_left=np.zeros((0, 2), dtype=float), bottom_right=np.zeros((0, 2), dtype=float))

    def add_point(self, point: tuple[float, float], label: Literal["top_left", "bottom_right"]) -> None:
        """add top left or bottom right point"""
        match label:
            case "top_left":
                self.top_left: Float[ndarray, "1 2"] = np.array([point], dtype=float)
            case "bottom_right":
                self.bottom_right: Float[ndarray, "1 2"] = np.array([point], dtype=float)

        if self.top_left.shape[0] == 1 and self.bottom_right.shape[0] == 1:
            # If both points are present, compute the bounding box
            self.bbox = np.array(
                [
                    self.top_left[0][0],
                    self.top_left[0][1],
                    self.bottom_right[0][0],
                    self.bottom_right[0][1],
                ],
                dtype=float,
            ).reshape(1, 4)

    def clear(self) -> None:
        """Clear all points"""
        self.top_left = np.zeros((0, 2), dtype=float)
        self.bottom_right = np.zeros((0, 2), dtype=float)


@dataclass
class CurrentState:
    recording_id: uuid.UUID
    rerun_log_paths: RerunLogPaths
    current_img_idx: int
    current_xyxy_idx: int
    current_uvc_idx: int
    radii: float


@serde
class SVPrediction:
    """A class to hold the prediction for a single image.

    Args:
        uvc_list: List of UVC coordinates for keypoints, where each element has shape (n_kpts, 3).
                 The last dimension contains [u, v, confidence].
        xyxy_list: List of bounding boxes in xyxy format, where each element has shape (4,).
                  Format is [x_min, y_min, x_max, y_max].
        rgb_hw3: RGB image array with shape (H, W, 3) and uint8 values.
        mask: Boolean mask array with shape (H, W) indicating segmented regions.
        depth_relative: Relative depth map with shape (H, W) and float values.
        pinhole_params: Camera pinhole parameters for the image.
    """

    seg_kpts: SAM2KeypointContainer
    uvc_list: list[Float[np.ndarray, "n_kpts 3"]]
    xyxy_list: list[XYXYContainer]
    info_message: str
    pinhole_params: PinholeParameters | None = None
    rgb_hw3: UInt8[ndarray, "H W 3"] | None = sfield(default=None, skip=True)
    mask: Bool[ndarray, "H W"] | None = sfield(default=None, skip=True)
    depth_relative: Float[ndarray, "H W"] | None = sfield(default=None, skip=True)
