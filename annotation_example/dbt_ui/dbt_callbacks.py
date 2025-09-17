from collections.abc import Generator
from dataclasses import replace

import gradio as gr
import numpy as np
import rerun as rr
from gradio_rerun.events import (
    SelectionChange,
    TimeUpdate,
)
from jaxtyping import Float
from rerun.event import SelectionChangeEvent

from annotation_example.dbt_ui.recording_utils import get_recording
from annotation_example.dbt_ui.state import AppState


def track_current_time(state: AppState, evt: TimeUpdate) -> AppState:
    timestamp_ns: float = evt.payload.time
    # if greater than 1e6 assume it's in nanoseconds and floor
    time_ns: int = int(np.floor(timestamp_ns))
    state: AppState = replace(state, current_time_ns=time_ns)
    return state


def _keypoint_entity_path(entity_path: str) -> str:
    return f"{entity_path}/annotations/manual_keypoint"


def _format_keypoint_status(
    keypoints: dict[str, dict[int, Float[np.ndarray, "n 2"]]],
    *,
    current_time_ns: int,
) -> str:
    if not keypoints:
        return "No keypoints yet."

    active_entries: list[tuple[str, Float[np.ndarray, "2"]]] = []
    for entity_path, by_time in keypoints.items():
        if current_time_ns in by_time:
            point: Float[np.ndarray, "2"] = by_time[current_time_ns][0]
            active_entries.append((entity_path, point))

    if not active_entries:
        return f"No keypoints at {current_time_ns * 1e-9:.3f}s."

    header: str = f"**Keypoints @ {current_time_ns * 1e-9:.3f}s**:"
    lines: list[str] = [header]
    for entity_path, point in sorted(active_entries):
        lines.append(f"- `{entity_path}` → ({point[0]:.1f}, {point[1]:.1f})")
    return "\n".join(lines)


def register_label_keypoint(
    state: AppState,
    _request: gr.Request,
    change: SelectionChange,
) -> Generator[tuple[bytes | None, AppState, str], None, None]:
    if not isinstance(state, AppState):
        yield None, state, "No keypoints yet."
        return

    current_time_ns: int = state.current_time_ns

    if state.active_control_panel != "Label":
        status: str = _format_keypoint_status(
            state.keypoints_by_entity_time,
            current_time_ns=current_time_ns,
        )
        yield None, state, status
        return

    if change is None:
        status = _format_keypoint_status(
            state.keypoints_by_entity_time,
            current_time_ns=current_time_ns,
        )
        yield None, state, status
        return

    evt: SelectionChangeEvent = change.payload

    if len(evt.items) != 1:
        status = _format_keypoint_status(
            state.keypoints_by_entity_time,
            current_time_ns=current_time_ns,
        )
        yield None, state, status
        return

    item = evt.items[0]
    if item.type != "entity" or item.position is None:
        status = _format_keypoint_status(
            state.keypoints_by_entity_time,
            current_time_ns=current_time_ns,
        )
        yield None, state, status
        return

    point_xy: Float[np.ndarray, "1 2"] = np.asarray([item.position[0:2]], dtype=np.float32)
    keypoints: dict[str, dict[int, Float[np.ndarray, "n 2"]]] = {
        entity_path: dict(points_by_time) for entity_path, points_by_time in state.keypoints_by_entity_time.items()
    }
    entity_points: dict[int, Float[np.ndarray, "n 2"]] = keypoints.get(item.entity_path, {})
    entity_points[current_time_ns] = point_xy
    keypoints[item.entity_path] = entity_points

    recording: rr.RecordingStream = get_recording(state.recording_id)
    stream: rr.BinaryStream = recording.binary_stream()

    target_entity_path: str = _keypoint_entity_path(item.entity_path)
    rr.set_time(
        state.rr_log_paths.timeline,
        duration=current_time_ns * 1e-9,
        recording=recording,
    )
    rr.log(
        target_entity_path,
        rr.Points2D(point_xy, colors=(255, 221, 0), radii=25),
        recording=recording,
    )

    stream.flush()
    payload: bytes = stream.read()
    new_state: AppState = replace(state, keypoints_by_entity_time=keypoints)
    status = _format_keypoint_status(
        new_state.keypoints_by_entity_time,
        current_time_ns=current_time_ns,
    )
    yield payload, new_state, status


def activate_run_networks_panel(state: AppState) -> AppState:
    updated_state: AppState = replace(state, active_control_panel="Run Networks")
    return updated_state


def activate_label_panel(state: AppState) -> AppState:
    updated_state: AppState = replace(state, active_control_panel="Label")
    return updated_state
