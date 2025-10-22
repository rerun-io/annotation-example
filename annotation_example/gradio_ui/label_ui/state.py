import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import numpy as np
from jaxtyping import Float, Int, UInt8
from numpy import ndarray
from rerun.event import SelectionChangeEvent
from simplecv.camera_parameters import PinholeParameters
from wilor_nano.hand_detection import DetectionResult


@dataclass
class RerunPaths:
    """Container for common Rerun entity paths used across the UI."""

    timeline: str = "video_time"
    """Timeline name used when logging time-series data into Rerun."""

    info_log_path: Path = Path("info")
    """Entity path for metadata and text logs."""

    parent_log_path: Path = Path("world")
    """Root entity under which new logs should be created."""

    ego_video_log_paths: list[Path] | None = None
    """Optional list of Rerun entities containing ego-vision video streams."""

    exo_video_log_paths: list[Path] | None = None
    """Optional list of Rerun entities containing exo-vision video streams."""


@dataclass()
class CurrentPrediction:
    """Latest detection results returned by the inference engine."""

    pinhole_params_list: list[PinholeParameters] | None = None

    detection_results: DetectionResult | None = None
    """Collection of detections for the current frame, if available."""


@dataclass
class BoundingBoxDraft:
    """In-progress bounding box annotation data."""

    top_left: Float[ndarray, "2"] | None = None
    """Coordinates of the top-left corner of the bounding box."""

    bottom_right: Float[ndarray, "2"] | None = None
    """Coordinates of the bottom-right corner of the bounding box."""

    xyxy: Float[ndarray, "1 4"] | None = None
    """Cached bounding box in XYXY format, if both corners are set."""

    ts_nano: int | None = None
    """Timestamp in nanoseconds when the bounding box was created."""

    entity_path: Path | None = None
    """Rerun entity path of the video stream that supplied the draft."""


@dataclass(slots=True, frozen=True)
class ConfirmedKeypointRecord:
    """Persisted keypoint annotations for a single frame."""

    entity_path: Path
    """Destination entity path for `Points2DWithConfidence` logs."""

    timeline: str
    """Timeline name used when logging the annotation."""

    timestamp_ns: int
    """Nanosecond timestamp corresponding to the annotation on `timeline`."""

    positions: Float[ndarray, "n_kpts 2"]
    """Full UV keypoint positions, including `NaN` placeholders."""

    confidences: Float[ndarray, "n_kpts"]
    """Confidence scores aligned with `positions`."""

    colors: UInt8[ndarray, "n_kpts 3"]
    """RGB colors derived from the confidences."""

    class_id: int
    """COCO 133 class identifier applied during logging."""

    keypoint_ids: tuple[int, ...]
    """Ordered COCO keypoint identifiers for the annotation."""


@dataclass(frozen=True)
class AppState:
    """Immutable top-level UI state shared between callbacks and panels."""

    recording_id: uuid.UUID
    """Unique identifier for the active Rerun recording."""

    rr_log_paths: RerunPaths = field(default_factory=RerunPaths)
    """Pre-configured Rerun paths to reuse while logging."""

    rrd_save_path: Path | None = None
    """Location where the generated RRD recording should be stored."""

    current_time_ns: int = 0
    """Current playback timestamp in nanoseconds."""

    current_tab: Literal["Info", "Annotations"] = "Info"
    """Active UI tab, used to control panel visibility."""

    shortest_timestamps: Int[ndarray, "n_frames"] | None = None
    """Per-frame timestamps shared across views, shortened for alignment."""

    video_timestamps_by_path: dict[str, Int[np.ndarray, "n_frames"]] = field(default_factory=dict)
    """Per-video frame timestamps keyed by Rerun entity path."""

    current_prediction: CurrentPrediction | None = None
    """Prediction data for the frame currently shown to the user."""

    selection_evt: SelectionChangeEvent | None = None
    """Latest selection event from the Rerun viewer, if any."""

    bounding_box_draft: BoundingBoxDraft = field(default_factory=BoundingBoxDraft)
    """Current in-progress bounding box annotation, if any."""

    active_video_entity_path: Path | None = None
    """Most recently interacted ego video entity path, used for clearing annotations."""

    confirmed_keypoints: dict[str, dict[int, ConfirmedKeypointRecord]] = field(default_factory=dict)
    """Cached hand annotations grouped by canonical ego path and timestamp."""
