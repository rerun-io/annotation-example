import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import gradio as gr
import numpy as np
import rerun as rr
from gradio_rerun.events import (
    SelectionChange,
)
from typing_extensions import TypedDict


def get_recording(recording_id) -> rr.RecordingStream:
    return rr.RecordingStream(application_id="multiview_sam_annotate", recording_id=recording_id)


class RerunLogPaths(TypedDict):
    timeline_name: str
    parent_log_path: Path
    cam_log_path_list: list[Path]


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


# In this function, the `request` and `evt` parameters will be automatically injected by Gradio when this event listener is fired.
#
# `SelectionChange` is a subclass of `EventData`: https://www.gradio.app/docs/gradio/eventdata
# `gr.Request`: https://www.gradio.app/main/docs/gradio/request
def update_keypoints(
    active_recording_id: uuid.UUID,
    point_type: Literal["include", "exclude"],
    mv_keypoint_dict: dict[str, KeypointsContainer],
    log_paths: RerunLogPaths,
    request: gr.Request,
    evt: SelectionChange,
):
    if active_recording_id == "":
        return

    # We can only log a keypoint if the user selected only a single item.
    if len(evt.items) != 1:
        return
    item = evt.items[0]

    # If the selected item isn't an entity, or we don't have its position, then bail out.
    if item.kind != "entity" or item.position is None:
        return

    # Now we can produce a valid keypoint.
    rec: rr.RecordingStream = get_recording(active_recording_id)
    stream: rr.BinaryStream = rec.binary_stream()
    current_keypoint: tuple[int, int] = item.position[0:2]

    for cam_name in mv_keypoint_dict:
        if cam_name in item.entity_path:
            # Update the keypoints for the specific camera
            mv_keypoint_dict[cam_name].add_point(current_keypoint, point_type)
            current_keypoint_container: KeypointsContainer = mv_keypoint_dict[cam_name]

    rec.set_time_nanos(log_paths["timeline_name"], nanos=0)
    # Log include points if any exist
    if current_keypoint_container.include_points.shape[0] > 0:
        rec.log(
            f"{item.entity_path}/include",
            rr.Points2D(current_keypoint_container.include_points, colors=(0, 255, 0), radii=5),
        )

    # Log exclude points if any exist
    if current_keypoint_container.exclude_points.shape[0] > 0:
        rec.log(
            f"{item.entity_path}/exclude",
            rr.Points2D(current_keypoint_container.exclude_points, colors=(255, 0, 0), radii=5),
        )

    # # Ensure we consume everything from the recording.
    stream.flush()
    yield stream.read(), mv_keypoint_dict
