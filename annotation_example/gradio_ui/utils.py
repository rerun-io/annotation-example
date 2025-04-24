import uuid
from dataclasses import dataclass
from typing import Literal

import numpy as np
import rerun as rr


def get_recording(recording_id: uuid.UUID | None, application_id: str = "Application ID") -> rr.RecordingStream:
    return rr.RecordingStream(application_id=application_id, recording_id=recording_id)


@dataclass
class KeypointsContainer:
    """Container for include and exclude keypoints"""

    include_points: np.ndarray  # shape (n,2)
    exclude_points: np.ndarray  # shape (m,2)

    @classmethod
    def empty(cls) -> "KeypointsContainer":
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
