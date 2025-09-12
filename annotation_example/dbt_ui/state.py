# label_app/state.py
import random
import uuid
from dataclasses import dataclass, field, replace
from enum import Enum, auto
from pathlib import Path
from typing import Literal

import numpy as np
from wilor_nano.hand_detection import DetectionResult


@dataclass
class RerunPaths:
    timeline: str = "video_time"
    info_log_path: Path = Path("info")
    parent_log_path: Path = Path("world")
    video_log_paths: list[Path] | None = None


@dataclass()
class CurrentPrediction:
    detection_results: DetectionResult | None = None


@dataclass(frozen=True, slots=True)
class AppState:
    recording_id: uuid.UUID
    rr_log_paths: RerunPaths = field(default_factory=RerunPaths)
    rrd_save_path: Path | None = None
    current_time_ns: int = 0
    current_tab: Literal["Info", "Annotations"] = "Info"
    # Path to the currently loaded video (if any)
    video_path: Path | None = None
    # Cached frame timestamps (ns) returned by AssetVideo.read_frame_timestamps_nanos()
    frame_timestamps_ns: np.ndarray | None = None
    current_prediction: CurrentPrediction | None = None


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
