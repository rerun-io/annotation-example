from typing import Literal

import gradio as gr
import numpy as np
import rerun as rr
from gradio_rerun.events import (
    SelectionChange,
)
from jaxtyping import Float, UInt8
from simplecv.rerun_log_utils import Points2DWithConfidence

from annotation_example.gradio_ui.person_annot_utils import (
    SELECTED_COLOR,
    CurrentState,
    RerunLogPaths,
    SVPrediction,
    XYXYContainer,
    get_recording,
)
from annotation_example.skeletons import COCO_17_IDS


def single_view_update_keypoints(
    # gradio components
    annot_mode: Literal["Box", "Keypoint", "Keypoint Segmentation"],
    point_type: Literal["include", "exclude"],
    xyxy_type: Literal["top_left", "bottom_right"],
    # gradio state
    state: CurrentState,
    sv_prediction: SVPrediction,
    request: gr.Request,
    change: SelectionChange,
):
    evt = change.payload

    # We can only log a keypoint if the user selected only a single item.
    if len(evt.items) != 1:
        return

    item = evt.items[0]

    # If the selected item isn't an entity, or we don't have its position, then bail out.
    if item.type != "entity" or item.position is None:
        return

    radii: float = state.radii
    rerun_log_paths: RerunLogPaths = state.rerun_log_paths

    # Now we can produce a valid keypoint.
    rec: rr.RecordingStream = get_recording(state.recording_id)
    stream: rr.BinaryStream = rec.binary_stream()
    selected_keypoint: tuple[int, int] = tuple(item.position[0:2])
    xyxy_container: XYXYContainer = sv_prediction.xyxy_list[state.current_xyxy_idx]

    match annot_mode:
        case "Box":
            xyxy_container.add_point(selected_keypoint, xyxy_type)
            rec.set_time(rerun_log_paths["timeline_name"], sequence=0)
            # Log include points if any exist
            if xyxy_container.top_left.shape[0] > 0:
                rec.log(
                    f"{rerun_log_paths['annotated_image_path']}/top_left_{state.current_xyxy_idx}",
                    rr.Points2D(xyxy_container.top_left, colors=(0, 0, 255), radii=radii, show_labels=True),
                )

            # Log exclude points if any exist
            if xyxy_container.bottom_right.shape[0] > 0:
                rec.log(
                    f"{rerun_log_paths['annotated_image_path']}/bottom_right{state.current_xyxy_idx}",
                    rr.Points2D(xyxy_container.bottom_right, colors=(255, 0, 0), radii=radii),
                )
            if xyxy_container.bottom_right.shape[0] > 0 and xyxy_container.top_left.shape[0] > 0:
                bbox = np.array(
                    [
                        xyxy_container.top_left[0][0],
                        xyxy_container.top_left[0][1],
                        xyxy_container.bottom_right[0][0],
                        xyxy_container.bottom_right[0][1],
                    ],
                    dtype=float,
                ).reshape(1, 4)
                rec.log(
                    f"{rerun_log_paths['annotated_image_path']}/bbox_{state.current_xyxy_idx}",
                    rr.Boxes2D(
                        array=bbox,
                        array_format=rr.Box2DFormat.XYXY,
                        labels=[f"Person {state.current_xyxy_idx}"],
                        colors=(255, 0, 0),
                        show_labels=True,
                    ),
                )
        case "Keypoint":
            rec.set_time(rerun_log_paths["timeline_name"], sequence=0)
            current_uvc: Float[np.ndarray, "n_kpts 3"] = sv_prediction.uvc_list[state.current_xyxy_idx]
            current_uv: Float[np.ndarray, "n_kpts 2"] = current_uvc[:, 0:2]
            current_confidence: Float[np.ndarray, "n_kpts 1"] = current_uvc[:, -1]
            # update current keypoint
            current_uv[state.current_uvc_idx, :] = np.array(selected_keypoint, dtype=float)
            colors: UInt8[np.ndarray, "n_kpts 3"] = np.full((current_uv.shape[0], 3), [0, 255, 0], dtype=np.uint8)
            # set the color of the current keypoint to WHITE
            colors[state.current_uvc_idx, :] = SELECTED_COLOR
            rec.log(
                f"{rerun_log_paths['annotated_image_path']}/keypoints_{state.current_xyxy_idx}",
                Points2DWithConfidence(
                    positions=current_uv,
                    confidences=current_confidence.squeeze(),
                    class_ids=2,
                    keypoint_ids=COCO_17_IDS,
                    colors=colors,
                    radii=radii,
                ),
            )
        case "Segmentation":
            raise NotImplementedError("Segmentation mode is not implemented in single_view_update_keypoints.")
            # keypoints_container.add_point(selected_keypoint, point_type)

            # rec.set_time(rerun_log_paths["timeline_name"], sequence=0)
            # # Log include points if any exist
            # if keypoints_container.include_points.shape[0] > 0:
            #     rec.log(
            #         f"{rerun_log_paths['annotated_image_path']}/include",
            #         rr.Points2D(keypoints_container.include_points, colors=(0, 255, 0), radii=radii),
            #     )

            # # Log exclude points if any exist
            # if keypoints_container.exclude_points.shape[0] > 0:
            #     rec.log(
            #         f"{rerun_log_paths['annotated_image_path']}/exclude",
            #         rr.Points2D(keypoints_container.exclude_points, colors=(255, 0, 0), radii=radii),
            #     )

    # Ensure we consume everything from the recording.
    stream.flush()
    yield stream.read()
