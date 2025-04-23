import uuid
from pathlib import Path
from typing import Literal

import gradio as gr
import rerun as rr
from gradio_rerun.events import SelectionChange
from typing_extensions import TypedDict

from annotation_example.gradio_ui.utils import KeypointsContainer, get_recording


class RerunLogPaths(TypedDict):
    timeline_name: str
    parent_log_path: Path
    cam_log_path_list: list[Path]


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
    change: SelectionChange,
):
    if active_recording_id == "":
        return

    evt = change.payload

    # We can only log a keypoint if the user selected only a single item.
    if len(evt.items) != 1:
        return
    item = evt.items[0]

    # If the selected item isn't an entity, or we don't have its position, then bail out.
    if item.type != "entity" or item.position is None:
        return

    # Now we can produce a valid keypoint.
    rec: rr.RecordingStream = get_recording(active_recording_id)
    stream: rr.BinaryStream = rec.binary_stream()
    current_keypoint: tuple[int, int] = tuple(item.position[0:2])

    for cam_name in mv_keypoint_dict:
        if cam_name in item.entity_path:
            # Update the keypoints for the specific camera
            mv_keypoint_dict[cam_name].add_point(current_keypoint, point_type)
            current_keypoint_container: KeypointsContainer = mv_keypoint_dict[cam_name]

    rec.set_time(log_paths["timeline_name"], sequence=0)
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
