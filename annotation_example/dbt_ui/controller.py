# label_app/controller.py
import uuid
from dataclasses import replace
from pathlib import Path
from typing import Literal, assert_never, cast

import cv2
import gradio as gr
import numpy as np
import open3d as o3d
import rerun as rr
import rerun.blueprint as rrb
from gradio_rerun.events import SelectionChangeEvent
from jaxtyping import Float, Int, UInt8
from natsort import natsorted
from numpy import ndarray
from rerun.event import ContainerSelectionItem, EntitySelectionItem, ViewSelectionItem
from simplecv.data.skeleton.mediapipe import MEDIAPIPE_ID2NAME, MEDIAPIPE_IDS, MEDIAPIPE_LINKS
from simplecv.rerun_log_utils import (
    Points2DWithConfidence,
    confidence_scores_to_rgb,
    log_pinhole,
    log_video,
)
from simplecv.video_io import MultiVideoReader
from wilor_nano.hand_detection import DetectionResult
from wilor_nano.hand_keypoints import KeypointResults

from annotation_example.dbt_ui.dbt_callbacks import (
    _format_keypoint_status,
    end_selection_processing,
)
from annotation_example.dbt_ui.engine import (
    HAND_CLASS_IDS,
    KEYPOINT_CONFIDENCE_THRESHOLD,
    Engine,
    MVCalibResults,
    time_to_frame_idx,
)
from annotation_example.dbt_ui.recording_utils import get_recording
from annotation_example.dbt_ui.state import AppState, CurrentPrediction, RerunPaths

HandName = Literal["left", "right"]
CornerName = Literal["top_left", "bottom_right"]

CORNER_SUFFIX_MAP: dict[CornerName, str] = {
    "top_left": "tl",
    "bottom_right": "br",
}

CORNER_TO_RADIO: dict[Literal["top_left", "bottom_right", "none"], str] = {
    "top_left": "TL",
    "bottom_right": "BR",
    "none": "None",
}

RADIO_TO_CORNER: dict[str, Literal["top_left", "bottom_right", "none"]] = {
    "top left": "top_left",
    "tl": "top_left",
    "bottom right": "bottom_right",
    "br": "bottom_right",
    "none": "none",
    "no bounding box": "none",
}

CORNER_LABEL_MAP: dict[CornerName, str] = {corner: suffix.upper() for corner, suffix in CORNER_SUFFIX_MAP.items()}

CORNER_COLOR_LOOKUP: dict[tuple[HandName, CornerName], tuple[int, int, int]] = {
    ("left", "top_left"): (0, 196, 255),
    ("left", "bottom_right"): (0, 128, 210),
    ("right", "top_left"): (255, 140, 0),
    ("right", "bottom_right"): (210, 32, 0),
}


def set_annotation_context(recording: rr.RecordingStream) -> None:
    rr.log(
        "/",
        rr.AnnotationContext(
            [
                rr.ClassDescription(
                    info=rr.AnnotationInfo(id=0, label="L", color=(0, 0, 255)),
                    keypoint_annotations=[
                        rr.AnnotationInfo(id=id, label=name) for id, name in MEDIAPIPE_ID2NAME.items()
                    ],
                    keypoint_connections=MEDIAPIPE_LINKS,
                ),
                rr.ClassDescription(
                    info=rr.AnnotationInfo(id=1, label="R", color=(255, 0, 0)),
                    keypoint_annotations=[
                        rr.AnnotationInfo(id=id, label=name) for id, name in MEDIAPIPE_ID2NAME.items()
                    ],
                    keypoint_connections=MEDIAPIPE_LINKS,
                ),
            ]
        ),
        static=True,
        recording=recording,
    )


def create_dbt_blueprint(
    recording: rr.RecordingStream, state: AppState, active_tab: Literal["Info", "Annotations"] = "Info"
) -> rrb.Blueprint:
    # Create a DBT blueprint for the given recording and app state.
    info_tab = rrb.TextDocumentView(name="Info")
    # Use a container for the Annotations tab so we can arrange multiple views.
    # NOTE: Tabs.active_tab only matches by name for View children, not Containers.
    #       When the child is a Container, we must select the tab by index instead.
    annotation_tab = rrb.Spatial3DView(line_grid=False)
    if state.rr_log_paths.ego_video_log_paths is not None:
        ego_2d_views: rrb.Vertical = rrb.Vertical(
            contents=[
                rrb.Spatial2DView(origin=video_log_path.parent)
                for video_log_path in state.rr_log_paths.ego_video_log_paths
            ]
        )
        annotation_tab = rrb.Horizontal(
            contents=[annotation_tab, ego_2d_views], name="Annotations", column_shares=[3, 1]
        )

    if state.rr_log_paths.exo_video_log_paths is not None:
        exo_2d_views: rrb.Horizontal = rrb.Horizontal(
            contents=[
                rrb.Spatial2DView(origin=video_log_path.parent)
                for video_log_path in state.rr_log_paths.exo_video_log_paths
            ]
        )
        annotation_tab = rrb.Vertical(contents=[annotation_tab, exo_2d_views], name="Annotations", row_shares=[3, 1])

    # Map requested tab name to index to support container child. If using directly it will fail to match.
    match active_tab:
        case "Info":
            active_idx = 0
        case "Annotations":
            active_idx = 1
        case _:
            assert_never(active_tab)
    blueprint: rrb.Blueprint = rrb.Blueprint(rrb.Tabs(info_tab, annotation_tab, active_tab=active_idx))
    return blueprint


def _extract_zip_to_videos_dir(zipfile_path: Path) -> Path:
    import zipfile

    # Strategy:
    # - If the zip has a single top-level directory, extract into the parent dir
    #   to avoid double-nesting (e.g. .../lg-videos/lg-videos/...).
    # - Otherwise, extract into a dedicated directory named after the zip stem.
    with zipfile.ZipFile(zipfile_path, "r") as zip_ref:
        members: list[str] = [m for m in zip_ref.namelist() if m and m.strip("/")]
        # Determine top-level entries
        top_levels = {m.split("/", 1)[0] for m in members}
        single_top: bool = len(top_levels) == 1
        top_name: str | None = next(iter(top_levels)) if single_top else None
        # Consider it a directory if there are entries like "top_name/..."
        is_dir_like: bool = single_top and any(m.startswith(f"{top_name}/") for m in members if m != top_name)

        if single_top and is_dir_like and top_name is not None:
            # Extract beside the zip (into parent) to avoid creating .../<stem>/<stem>/...
            target_base: Path = zipfile_path.parent
            print(f"Extracting {zipfile_path} to {target_base} (preserving top-level folder '{top_name}')")
            zip_ref.extractall(target_base)
            videos_dir: Path = target_base / top_name
        else:
            # Mixed contents or a single file at top-level: extract into a dedicated dir
            extract_dir: Path = zipfile_path.parent / zipfile_path.stem
            print(f"Extracting {zipfile_path} to {extract_dir}")
            extract_dir.mkdir(parents=True, exist_ok=True)
            zip_ref.extractall(extract_dir)
            videos_dir = extract_dir

    assert videos_dir.exists(), f"Expected {videos_dir} to exist after extraction."
    print(f"Extracted to: {videos_dir}")
    return videos_dir


class Controller:
    """Glue between UI events, Engine calls, pure state transitions, and Rerun logging."""

    def __init__(
        self,
        engine: Engine,
    ):
        self.engine: Engine = engine
        # only show info on the start, so we'll switch to Annotations tab on video upload
        self.tab_name: Literal["Info", "Annotations"] = "Info"

    def set_hand_selection(
        self,
        state: gr.State | AppState,
        hand_label: str | None,
    ) -> AppState | gr.State:
        if not isinstance(state, AppState):
            return state

        label_input: str = (hand_label or "").strip().lower()
        mapping: dict[str, Literal["left", "right"]] = {
            "left hand": "left",
            "right hand": "right",
            "left": "left",
            "right": "right",
        }
        selected_hand: Literal["left", "right"] = mapping.get(label_input, state.selected_hand)
        updated_state: AppState = replace(state, selected_hand=selected_hand)
        return updated_state

    def set_bbox_corner_selection(
        self,
        state: gr.State | AppState,
        corner_label: str | None,
    ) -> AppState | gr.State:
        if not isinstance(state, AppState):
            return state

        label_input: str = (corner_label or "").strip().lower()
        selected_corner: Literal["top_left", "bottom_right", "none"] = RADIO_TO_CORNER.get(
            label_input, state.selected_bbox_corner
        )
        updated_state: AppState = replace(state, selected_bbox_corner=selected_corner)
        return updated_state

    def on_corner_selection_changed(self, state):
        yield from self._on_corner_selection_changed(state)

    def _on_corner_selection_changed(self, state: gr.State | AppState):
        if not isinstance(state, AppState):
            yield None, state, "No keypoints yet.", CORNER_TO_RADIO["top_left"]
            return

        if state.selected_bbox_corner != "none":
            status: str = _format_keypoint_status(
                state.keypoints_by_entity_time,
                current_time_ns=state.current_time_ns,
            )
            yield None, state, status, CORNER_TO_RADIO[state.selected_bbox_corner]
            return

        yield from self._clear_hand_annotations(state)

    # ---- handlers wired by the panel ----
    def log_state(self, state):
        yield from self._log_state(state)

    def _log_state(self, state: gr.State | AppState):
        """Log the current image to Rerun."""
        recording: rr.RecordingStream = get_recording(state.recording_id)
        blueprint: rrb.Blueprint = create_dbt_blueprint(recording, state, active_tab=self.tab_name)
        rr.send_blueprint(blueprint, recording=recording)
        rr.log("info", rr.TextDocument("Press 'Next' to start annotating!"), static=True, recording=recording)

        # log the current prediction if it exists
        if state.current_prediction is not None:
            # set the timeline
            rr.set_time(
                timeline=state.rr_log_paths.timeline, duration=state.current_time_ns * 1e-9, recording=recording
            )
            print("Logging current prediction to Rerun")
            current_pred: CurrentPrediction = state.current_prediction
            if current_pred.detection_results.right_xyxy is not None:
                rr.log(
                    f"{state.rr_log_paths.parent_log_path}/video/right_xyxy",
                    rr.Boxes2D(array=current_pred.detection_results.right_xyxy, array_format=rr.Box2DFormat.XYXY),
                    recording=recording,
                )
            if current_pred.detection_results.left_xyxy is not None:
                rr.log(
                    f"{state.rr_log_paths.parent_log_path}/video/left_xyxy",
                    rr.Boxes2D(array=current_pred.detection_results.left_xyxy, array_format=rr.Box2DFormat.XYXY),
                    recording=recording,
                )

        yield recording.binary_stream().read(), state

    def log_keypoint_clicks(self, state):
        """Log keypoints to Rerun when the user clicks inside a 2D view."""

        yield from self._log_keypoint_clicks(state)

    def _log_keypoint_clicks(self, state: gr.State | AppState):
        """Persist the selected point and update UI state/status messaging."""

        if not isinstance(state, AppState):
            yield None, state, "No keypoints yet.", CORNER_TO_RADIO["top_left"]
            return

        recording_id = state.recording_id

        try:
            if state.selection_evt is None:
                yield None, state, "No keypoints yet.", CORNER_TO_RADIO[state.selected_bbox_corner]
                return

            current_time_ns: int = state.current_time_ns
            evt: SelectionChangeEvent = state.selection_evt
            items: list[EntitySelectionItem | ViewSelectionItem | ContainerSelectionItem] = evt.items
            if not (len(items) == 1 and isinstance(items[0], EntitySelectionItem)):
                raise gr.Error("Please select a single entity to log a keypoint.")
            item: EntitySelectionItem = items[0]
            entity_path: Path = Path(item.entity_path)
            # make sure that we're only logging keypoints on pinhole cameras
            pinhole_path: Path = entity_path.parent
            if pinhole_path.name != "pinhole" and entity_path.name != "video":
                yield (
                    None,
                    state,
                    f"Selected entity is not a pinhole camera: {item.entity_path}",
                    CORNER_TO_RADIO[state.selected_bbox_corner],
                )
                return

            point_xy: Float[np.ndarray, "1 2"] = np.asarray([item.position[0:2]], dtype=np.float32)
            selected_hand: Literal["left", "right"] = state.selected_hand
            selected_corner: Literal["top_left", "bottom_right", "none"] = state.selected_bbox_corner
            if selected_corner == "none":
                # Treat a click while "None" is selected as a request to clear annotations for this view.
                yield from self._clear_hand_annotations(state)
                return

            corner_suffix: str = CORNER_SUFFIX_MAP[selected_corner]
            target_entity_str: str = (pinhole_path / selected_hand / f"{corner_suffix}_xyxy_kp").as_posix()
            colors: tuple[int, int, int] = CORNER_COLOR_LOOKUP[(selected_hand, selected_corner)]

            keypoints: dict[str, dict[int, Float[np.ndarray, "n 2"]]] = {
                str(entity_path): dict(points_by_time)
                for entity_path, points_by_time in state.keypoints_by_entity_time.items()
            }
            entity_points: dict[int, Float[np.ndarray, "n 2"]] = keypoints.get(target_entity_str, {})
            entity_points[current_time_ns] = point_xy
            keypoints[target_entity_str] = entity_points

            recording: rr.RecordingStream = get_recording(state.recording_id)
            stream: rr.BinaryStream = recording.binary_stream()

            rr.set_time(
                state.rr_log_paths.timeline,
                duration=current_time_ns * 1e-9,
                recording=recording,
            )
            rr.log(
                target_entity_str,
                rr.Points2D(
                    point_xy,
                    colors=colors,
                    radii=15,
                    labels=CORNER_LABEL_MAP[selected_corner],
                ),
                recording=recording,
            )

            stream.flush()
            payload_bytes: bytes = stream.read()
            payload: bytes | None = payload_bytes if payload_bytes else None
            next_corner: Literal["top_left", "bottom_right"] = (
                "bottom_right" if selected_corner == "top_left" else "top_left"
            )
            new_state: AppState = replace(
                state,
                keypoints_by_entity_time=keypoints,
                selected_bbox_corner=next_corner,
                selection_evt=None,
            )
            status: str = _format_keypoint_status(
                new_state.keypoints_by_entity_time,
                current_time_ns=current_time_ns,
            )
            next_radio_value: str = CORNER_TO_RADIO[next_corner]
            yield payload, new_state, status, next_radio_value
        finally:
            end_selection_processing(recording_id)
        return

    def confirm_bounding_boxes(self, state):
        yield from self._confirm_bounding_boxes(state)

    def _confirm_bounding_boxes(self, state: gr.State | AppState):
        if not isinstance(state, AppState):
            yield None, state, "No keypoints yet.", CORNER_TO_RADIO["top_left"]
            return

        current_time_ns: int = state.current_time_ns
        keypoints_snapshot: dict[str, dict[int, Float[np.ndarray, "n 2"]]] = {
            entity_path: dict(points_by_time) for entity_path, points_by_time in state.keypoints_by_entity_time.items()
        }

        # Collect TL/BR pairs per hand and pinhole.
        corner_points: dict[tuple[str, HandName], dict[str, tuple[str, Float[np.ndarray, "1 2"]]]] = {}
        for path_str, by_time in keypoints_snapshot.items():
            if current_time_ns not in by_time:
                continue
            path_obj: Path = Path(path_str)
            corner_name: str = path_obj.name
            if corner_name not in {"tl_xyxy_kp", "br_xyxy_kp"}:
                continue
            hand_name: str = path_obj.parent.name
            if hand_name not in {"left", "right"}:
                continue
            pinhole_path: Path = path_obj.parent.parent
            hand_literal: HandName = cast(HandName, hand_name)
            corner_key: CornerName = "top_left" if corner_name.startswith("tl") else "bottom_right"
            corner_points.setdefault((pinhole_path.as_posix(), hand_literal), {})[corner_key] = (
                path_str,
                by_time[current_time_ns],
            )

        if not corner_points:
            status: str = _format_keypoint_status(
                state.keypoints_by_entity_time,
                current_time_ns=current_time_ns,
            )
            yield None, state, "No corners to confirm.", CORNER_TO_RADIO[state.selected_bbox_corner]
            return

        recording: rr.RecordingStream = get_recording(state.recording_id)
        stream: rr.BinaryStream = recording.binary_stream()
        timeline: str = state.rr_log_paths.timeline

        updated_keypoints: dict[str, dict[int, Float[np.ndarray, "n 2"]]] = {
            entity_path: dict(points_by_time) for entity_path, points_by_time in state.keypoints_by_entity_time.items()
        }
        cleared_paths: set[str] = set()
        confirmed_labels: list[str] = []
        missing_pairs: list[str] = []
        camera_lookup_failures: list[str] = []

        for (pinhole_path_str, hand_name), corners in corner_points.items():
            missing = {"top_left", "bottom_right"} - corners.keys()
            if missing:
                labels = ", ".join(sorted(missing))
                missing_pairs.append(f"{hand_name} ({Path(pinhole_path_str).name}) missing {labels}")
                continue
            tl_path, tl_xy = corners["top_left"]
            br_path, br_xy = corners["bottom_right"]
            tl_coords: Float[np.ndarray, "1 2"] = tl_xy
            br_coords: Float[np.ndarray, "1 2"] = br_xy
            x1, y1 = tl_coords[0]
            x2, y2 = br_coords[0]
            x_min: float = float(min(x1, x2))
            x_max: float = float(max(x1, x2))
            y_min: float = float(min(y1, y2))
            y_max: float = float(max(y1, y2))
            box_xyxy: Float[np.ndarray, "1 4"] = np.asarray([[x_min, y_min, x_max, y_max]], dtype=np.float32)

            pinhole_path = Path(pinhole_path_str)
            hand_literal: HandName = cast(HandName, hand_name)
            hand_path: Path = pinhole_path / hand_literal

            cam_type: Literal["ego", "exo"] = "ego" if "ego" in pinhole_path.parts else "exo"
            if cam_type == "ego":
                mv_reader: MultiVideoReader | None = self.engine.ego_mv_reader
            else:
                mv_reader = self.engine.exo_mv_reader

            if mv_reader is None:
                camera_lookup_failures.append(f"Missing {cam_type} video reader for {pinhole_path.name}")
                continue

            ts_nano: int = state.current_time_ns
            video_entity_path: Path = pinhole_path / "video"
            camera_timestamps: Int[ndarray, "n_frames"] | None = state.video_timestamps_by_path.get(
                video_entity_path.as_posix()
            )

            ts_idx: int = -1
            if camera_timestamps is not None and camera_timestamps.size > 0:
                ts_idx = time_to_frame_idx(ts_nano, camera_timestamps)
            else:
                all_ts_nano: Int[ndarray, "n_frames"] = state.shortest_timestamps
                if all_ts_nano is not None and all_ts_nano.size > 0:
                    ts_idx = time_to_frame_idx(ts_nano, all_ts_nano)

            if ts_idx < 0:
                camera_lookup_failures.append(f"Missing timestamp alignment for {video_entity_path.as_posix()}")
                continue

            frame_count: int = len(mv_reader)
            if frame_count == 0:
                camera_lookup_failures.append(f"No frames available for {video_entity_path.as_posix()}")
                continue
            if ts_idx >= frame_count:
                camera_lookup_failures.append(
                    f"Timestamp {ts_nano} clamped to final frame for {video_entity_path.as_posix()}"
                )
                ts_idx = frame_count - 1

            bgr_list = mv_reader[ts_idx]
            rgb_list: list[UInt8[ndarray, "h w 3"]] = [cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB) for bgr in bgr_list]

            camera_part: str | None = next((part for part in pinhole_path.parts if part.startswith("camera_")), None)
            if camera_part is None:
                camera_lookup_failures.append(f"Unable to infer camera id from {pinhole_path.as_posix()}")
                continue

            try:
                camera_idx: int = int(camera_part.split("_")[1])
            except (IndexError, ValueError):
                camera_lookup_failures.append(f"Malformed camera identifier '{camera_part}'")
                continue

            if camera_idx >= len(rgb_list) or camera_idx < 0:
                camera_lookup_failures.append(f"Camera index {camera_idx} out of range for {pinhole_path.as_posix()}")
                continue

            rgb_hw3: UInt8[ndarray, "h w 3"] = rgb_list[camera_idx]

            kpts_results: KeypointResults = self.engine.hand_keypoint_engine(
                rgb_hw3=rgb_hw3, xyxy=box_xyxy, handedness=hand_literal
            )
            uv: Float[ndarray, "n_frames=1 n_kpts=21 2"] = kpts_results.keypoints_2d
            conf: Float[ndarray, "n_frames=1 n_kpts=21"] = kpts_results.scores
            conf_colors: UInt8[ndarray, "n_frames=1 n_kpts=21 3"] = confidence_scores_to_rgb(
                confidence_scores=conf[..., np.newaxis]
            )
            conf_values: Float[np.ndarray, "n_kpts"] = conf[0].astype(np.float32)
            valid_mask: np.ndarray = conf_values >= KEYPOINT_CONFIDENCE_THRESHOLD
            uv_filtered: Float[np.ndarray, "n_kpts 2"] = uv[0].copy()
            uv_filtered[~valid_mask] = np.nan

            rr.set_time(timeline, duration=current_time_ns * 1e-9, recording=recording)
            # Overwrite the same entity used by automatic detections so manual confirmation replaces it.
            rr.log(
                f"{hand_path.as_posix()}_xyxy",
                rr.Boxes2D(
                    array=box_xyxy,
                    array_format=rr.Box2DFormat.XYXY,
                    class_ids=HAND_CLASS_IDS[hand_literal],
                    show_labels=True,
                ),
                recording=recording,
            )
            rr.log(
                f"{state.rr_log_paths.parent_log_path}/video/{hand_literal}_xyxy",
                rr.Boxes2D(
                    array=box_xyxy,
                    array_format=rr.Box2DFormat.XYXY,
                    class_ids=HAND_CLASS_IDS[hand_literal],
                    show_labels=False,
                ),
                recording=recording,
            )
            rr.log(
                f"{hand_path}_keypoints",
                Points2DWithConfidence(
                    positions=uv_filtered,
                    confidences=conf_values,
                    class_ids=HAND_CLASS_IDS[hand_literal],
                    keypoint_ids=MEDIAPIPE_IDS,
                    show_labels=False,
                    colors=conf_colors[0],
                ),
                recording=recording,
            )
            # Clear the temporary keypoint markers and drop them from state.
            for _corner_key, (entity_path_str, _) in corners.items():
                rr.log(entity_path_str, rr.Clear(recursive=False), recording=recording)
                cleared_paths.add(entity_path_str)
                points_by_time = updated_keypoints.get(entity_path_str)
                if points_by_time and current_time_ns in points_by_time:
                    del points_by_time[current_time_ns]
                    if not points_by_time:
                        updated_keypoints.pop(entity_path_str, None)

            confirmed_labels.append(f"{hand_literal} ({pinhole_path.name})")

        # Re-log remaining keypoints for cleared entities at other timestamps.
        for path_str in cleared_paths:
            remaining = updated_keypoints.get(path_str)
            if not remaining:
                continue
            path_obj: Path = Path(path_str)
            hand_name: HandName = cast(HandName, path_obj.parent.name)
            corner_name: CornerName = "top_left" if path_obj.name.startswith("tl") else "bottom_right"
            label: str = CORNER_LABEL_MAP[corner_name]
            colors: tuple[int, int, int] = CORNER_COLOR_LOOKUP[(hand_name, corner_name)]

            for timestamp_ns, points in remaining.items():
                rr.set_time(timeline, duration=timestamp_ns * 1e-9, recording=recording)
                rr.log(
                    path_str,
                    rr.Points2D(points, colors=colors, radii=15, labels=label),
                    recording=recording,
                )

        stream.flush()
        payload_bytes: bytes = stream.read()
        payload: bytes | None = payload_bytes if payload_bytes else None

        new_state: AppState = replace(
            state,
            keypoints_by_entity_time=updated_keypoints,
            selected_bbox_corner="top_left",
            selection_evt=None,
        )

        if state.current_prediction is not None and state.current_prediction.detection_results is not None:
            det: DetectionResult = state.current_prediction.detection_results
            det = (
                replace(det, left_xyxy=box_xyxy) if state.selected_hand == "left" else replace(det, right_xyxy=box_xyxy)
            )
            new_pred: CurrentPrediction = replace(state.current_prediction, detection_results=det)
            new_state = replace(new_state, current_prediction=new_pred)
        status: str = _format_keypoint_status(
            new_state.keypoints_by_entity_time,
            current_time_ns=current_time_ns,
        )
        error_messages: list[str] = []
        if missing_pairs:
            error_messages.append(f"Need TL+BR before confirming: {', '.join(missing_pairs)}")
        if camera_lookup_failures:
            error_messages.append(f"Camera issues: {', '.join(camera_lookup_failures)}")

        if confirmed_labels:
            status = f"Confirmed boxes for: {', '.join(confirmed_labels)}"
        elif error_messages:
            status = " | ".join(error_messages)

        yield payload, new_state, status, CORNER_TO_RADIO[new_state.selected_bbox_corner]

    def _clear_hand_annotations(self, state: AppState):
        selected_hand: HandName = state.selected_hand
        current_time_ns: int = state.current_time_ns
        evt: SelectionChangeEvent | None = state.selection_evt
        if evt is None or not evt.items:
            status: str = _format_keypoint_status(
                state.keypoints_by_entity_time,
                current_time_ns=current_time_ns,
            )
            yield None, state, "Select a view before clearing.", CORNER_TO_RADIO[state.selected_bbox_corner]
            return

        item = evt.items[0]
        if not isinstance(item, EntitySelectionItem):
            status: str = _format_keypoint_status(
                state.keypoints_by_entity_time,
                current_time_ns=current_time_ns,
            )
            yield None, state, "Select a video before clearing.", CORNER_TO_RADIO[state.selected_bbox_corner]
            return

        entity_path: Path = Path(item.entity_path)
        if entity_path.name != "video":
            status: str = _format_keypoint_status(
                state.keypoints_by_entity_time,
                current_time_ns=current_time_ns,
            )
            yield None, state, "Select a video before clearing.", CORNER_TO_RADIO[state.selected_bbox_corner]
            return

        pinhole_path: Path = entity_path.parent
        hand_path: Path = pinhole_path / selected_hand

        recording: rr.RecordingStream = get_recording(state.recording_id)
        stream: rr.BinaryStream = recording.binary_stream()
        rr.set_time(state.rr_log_paths.timeline, duration=current_time_ns * 1e-9, recording=recording)

        updated_keypoints: dict[str, dict[int, Float[np.ndarray, "n 2"]]] = {
            entity_path: dict(points_by_time) for entity_path, points_by_time in state.keypoints_by_entity_time.items()
        }

        def remove_manual_entry(path_str: str) -> None:
            points_by_time = updated_keypoints.get(path_str)
            if points_by_time is None:
                return
            if current_time_ns in points_by_time:
                del points_by_time[current_time_ns]
            if not points_by_time:
                updated_keypoints.pop(path_str, None)

        for manual_suffix in ("tl_xyxy_kp", "br_xyxy_kp"):
            manual_path: str = (hand_path / manual_suffix).as_posix()
            remove_manual_entry(manual_path)
            rr.log(manual_path, rr.Clear(recursive=False), recording=recording)

        rr.log(f"{hand_path.as_posix()}_xyxy", rr.Clear(recursive=False), recording=recording)
        rr.log(f"{hand_path.as_posix()}_keypoints", rr.Clear(recursive=False), recording=recording)
        rr.log(
            f"{state.rr_log_paths.parent_log_path}/video/{selected_hand}_xyxy",
            rr.Clear(recursive=False),
            recording=recording,
        )

        stream.flush()
        payload_bytes: bytes = stream.read()
        payload: bytes | None = payload_bytes if payload_bytes else None

        new_state: AppState = replace(
            state,
            keypoints_by_entity_time=updated_keypoints,
            selected_bbox_corner="top_left",
            selection_evt=None,
        )

        if state.current_prediction is not None and state.current_prediction.detection_results is not None:
            det: DetectionResult = state.current_prediction.detection_results
            det = replace(det, left_xyxy=None) if selected_hand == "left" else replace(det, right_xyxy=None)
            new_pred: CurrentPrediction = replace(state.current_prediction, detection_results=det)
            new_state = replace(new_state, current_prediction=new_pred)

        status: str = _format_keypoint_status(
            new_state.keypoints_by_entity_time,
            current_time_ns=current_time_ns,
        )

        yield payload, new_state, status, CORNER_TO_RADIO[new_state.selected_bbox_corner]

    def initialize_rrd(self, video, state):
        yield from self._initialize_rrd(video, state)

    def _initialize_rrd(self, zip_path: str | None, state: gr.State | AppState):
        # switch to Annotations tab on video upload
        self.tab_name = "Annotations"
        recording: rr.RecordingStream = get_recording(state.recording_id)
        stream: rr.BinaryStream = recording.binary_stream()
        set_annotation_context(recording=recording)

        if zip_path is None:
            # create a new recording id to clear out any previous video state
            state = AppState(recording_id=uuid.uuid4())
            # remove video readers from the engine
            self.engine.ego_mv_reader = None
            self.engine.exo_mv_reader = None
            yield None, state
            return

        zip_path = Path(zip_path)
        videos_dir: Path = _extract_zip_to_videos_dir(zip_path)
        # get the subfolers of videos dir
        subfolders: list[Path] = [f for f in videos_dir.iterdir() if f.is_dir()]
        # Check for presence of 'ego' or 'exo' subfolders
        ego_present: bool = any(f.name == "ego" for f in subfolders)
        exo_present: bool = any(f.name == "exo" for f in subfolders)

        if not (ego_present or exo_present):
            raise gr.Error("The uploaded zip must contain 'ego' and/or 'exo' subfolders with videos.")

        timeline_candidates: list[Int[ndarray, "num_frames"]] = []
        if ego_present:
            result: tuple[AppState, list[Int[ndarray, "num_frames"]]] = self._log_video_group(
                group="ego", videos_dir=videos_dir, state=state, recording=recording
            )
            state, ego_timestamps = result
            timeline_candidates.extend(ego_timestamps)
        if exo_present:
            result: tuple[AppState, list[Int[ndarray, "num_frames"]]] = self._log_video_group(
                group="exo", videos_dir=videos_dir, state=state, recording=recording
            )
            state, exo_timestamps = result
            timeline_candidates.extend(exo_timestamps)

        # check for the shortest video timestamps to use as the main timeline
        shortest_timestamps: Int[ndarray, "n_frames"] = min(timeline_candidates, key=len)
        state: AppState = replace(state, shortest_timestamps=shortest_timestamps)

        blueprint: rrb.Blueprint = create_dbt_blueprint(recording, state, active_tab=self.tab_name)
        rr.send_blueprint(blueprint, recording=recording)

        yield stream.read(), state

    def _log_video_group(
        self,
        *,
        group: Literal["ego", "exo"],
        videos_dir: Path,
        state: AppState,
        recording: rr.RecordingStream,
    ) -> tuple[AppState, list[Int[ndarray, "num_frames"]]]:
        group_dir: Path = videos_dir / group
        if not group_dir.exists():
            raise gr.Error(f"Video path {group_dir} does not exist!")

        video_paths: list[Path] = list(natsorted(group_dir.glob("*.mp4")))
        if not video_paths:
            raise gr.Error("No videos found in uploaded zip")

        video_log_paths: list[Path] = []
        timestamps_list: list[Int[ndarray, "num_frames"]] = []
        timestamps_by_path: dict[str, Int[ndarray, "num_frames"]] = dict(state.video_timestamps_by_path)
        for index, video_path in enumerate(video_paths):
            video_log_path: Path = state.rr_log_paths.parent_log_path / group / f"camera_{index}" / "pinhole" / "video"
            frame_timestamps_ns: Int[ndarray, "num_frames"] = log_video(
                video_path=video_path,
                video_log_path=video_log_path,
                timeline=state.rr_log_paths.timeline,
                recording=recording,
            )
            video_log_paths.append(video_log_path)
            timestamps_list.append(frame_timestamps_ns)
            timestamps_by_path[video_log_path.as_posix()] = frame_timestamps_ns

        log_path_update: dict[str, list[Path]] = {f"{group}_video_log_paths": video_log_paths}
        updated_rr_paths: RerunPaths = replace(state.rr_log_paths, **log_path_update)
        state: AppState = replace(
            state,
            rr_log_paths=updated_rr_paths,
            video_timestamps_by_path=timestamps_by_path,
        )

        mv_reader: MultiVideoReader = MultiVideoReader(video_paths)
        setattr(self.engine, f"{group}_mv_reader", mv_reader)

        return state, timestamps_list

    def log_calibration_results(
        self,
        state,
        progress=gr.Progress(track_tqdm=True),
    ):
        yield from self._log_calibration_results(state, progress)

    def _log_calibration_results(
        self,
        state: gr.State | AppState,
        progress: gr.Progress,
    ):
        recording: rr.RecordingStream = get_recording(state.recording_id)
        stream: rr.BinaryStream = recording.binary_stream()

        progress(0.0, desc="Starting multiview calibration…")

        bgr_list: list[UInt8[ndarray, "H W 3"]] = self.engine.exo_mv_reader[0]
        rgb_list: list[UInt8[ndarray, "H W 3"]] = [cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB) for bgr in bgr_list]
        mv_results: MVCalibResults = self.engine.calibrate_mv(state=state, rgb_list=rgb_list)

        progress(0.5, desc="Logging calibration results…")

        pcd_ds: o3d.geometry.PointCloud = mv_results.pcd
        # log the pointcloud
        filtered_points: Float[ndarray, "final_points 3"] = np.asarray(pcd_ds.points, dtype=np.float32)
        filtered_colors: Float[ndarray, "final_points 3"] = np.asarray(pcd_ds.colors, dtype=np.float32)

        rr.log(
            f"{state.rr_log_paths.parent_log_path}/point_cloud",
            rr.Points3D(
                filtered_points,
                colors=filtered_colors,
            ),
            static=True,
            recording=recording,
        )

        # log the cameras
        video_log_paths = state.rr_log_paths.exo_video_log_paths
        cam_log_paths: list[Path] = [video_log_path.parent.parent for video_log_path in video_log_paths]
        for pinhole_param, cam_log_path in zip(mv_results.pinhole_param_list, cam_log_paths, strict=True):
            log_pinhole(
                pinhole_param,
                cam_log_path=cam_log_path,
                static=True,
                image_plane_distance=0.1,
                recording=recording,
            )
        # update the current prediction with the pinhole params
        if state.current_prediction is not None:
            raise gr.Error("Current prediction should be None before calibration.")
        current_pred: CurrentPrediction = CurrentPrediction(pinhole_params_list=mv_results.pinhole_param_list)
        state = replace(state, current_prediction=current_pred)

        yield stream.read(), state

    # def on_nav(self, state: AppState, kind: Action) -> AppState:
    #     """Handle navigation events."""
    #     new_state: AppState = reduce(state, kind)
    #     return new_state
