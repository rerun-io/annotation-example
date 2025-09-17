# label_app/state.py
import random
import uuid
from dataclasses import dataclass, field, replace
from enum import Enum, auto
from pathlib import Path
from typing import Literal

import numpy as np
from jaxtyping import Float, Int
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

    current_prediction: CurrentPrediction | None = None
    """Prediction data for the frame currently shown to the user."""

    selection_evt: SelectionChangeEvent | None = None
    """Latest selection event from the Rerun viewer, if any."""

    keypoints_by_entity_time: dict[str, dict[int, Float[np.ndarray, "n 2"]]] = field(default_factory=dict)
    """Manual 2D keypoints logged per entity path and timestamp (ns)."""

    active_control_panel: Literal["Run Networks", "Label"] = "Run Networks"
    """Left-hand control tab currently selected by the user."""

    selected_hand: Literal["left", "right"] = "left"
    """Currently targeted hand for manual bounding-box annotation."""

    selected_bbox_corner: Literal["top_left", "bottom_right", "none"] = "none"
    """Active bounding-box corner selection used while collecting manual clicks."""


class Action(Enum):
    NEXT = auto()
    PREV = auto()
    RAND = auto()
    SKIP = auto()


def reduce(state: AppState, action: Action) -> AppState:
    state = replace(state, recording_id=uuid.uuid4())  # ensure immutability
    match action:
        case Action.NEXT:
            return replace(state, idx=min(state.idx + 1, len(state.images) - 1))
        case Action.PREV:
            return replace(state, idx=max(0, state.idx - 1))
        case Action.RAND:
            return replace(state, idx=random.randint(0, len(state.images) - 1))
        case Action.SKIP:
            return replace(state, idx=(state.idx + 1) % len(state.images))
        case _:
            raise ValueError(f"Unknown action: {action}")
