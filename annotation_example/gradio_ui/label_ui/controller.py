import tempfile
import uuid
from dataclasses import replace
from pathlib import Path
from subprocess import CalledProcessError, run
from typing import Literal

import gradio as gr
import numpy as np
import rerun as rr
import rerun.blueprint as rrb
from jaxtyping import Float, Int, UInt8
from numpy import ndarray
from rerun.dataframe import Recording
from rerun.event import EntitySelectionItem, SelectionChangeEvent
from simplecv.apis.view_exoego import (
    SceneSetupResult,
    create_container,
    filter_out_of_bounds_keypoints,
    log_environment_mesh,
    log_exoego_batch,
    setup_scene,
)
from simplecv.camera_parameters import PinholeParameters
from simplecv.data.exoego.base_exoego import BaseExoEgoSequence
from simplecv.data.exoego.rrd_exoego import RRDExoEgoConfig
from simplecv.data.skeleton.coco_133 import COCO_133_IDS, LEFT_HAND_IDX, RIGHT_HAND_IDX
from simplecv.rerun_custom_types import Points2DWithConfidence, confidence_scores_to_rgb
from simplecv.rerun_log_utils import RerunTyroConfig

from mv_api.api.full_exoego_pipeline import set_annotation_context
from mv_api.coco133_layers import COCO133_PREDICTION_LAYER_TO_PATH, Coco133AnnotationLayer
from mv_api.gradio_ui.label_ui.engine import Engine
from mv_api.gradio_ui.label_ui.recording_utils import get_recording
from mv_api.gradio_ui.label_ui.state import AppState, BoundingBoxDraft, ConfirmedKeypointRecord


class Controller:
    """Glue between UI events, Engine calls, pure state transitions, and Rerun logging."""

    def __init__(
        self,
        engine: Engine,
    ):
        self.engine: Engine = engine
        self._confirmed_keypoints: dict[
            tuple[str, int],
            tuple[Float[ndarray, "n_kpts 2"], Float[ndarray, "n_kpts"]],
        ] = {}
        self._projected_keypoints_cache: dict[
            str,
            tuple[Float[ndarray, "n_frames n_kpts 2"], Float[ndarray, "n_frames n_kpts"]],
        ] = {}
        self._current_sequence: BaseExoEgoSequence | None = None
        self._source_rrd_path: Path | None = None
        # only show info on the start, so we'll switch to Annotations tab on video upload
        self.tab_name: Literal["Info", "Annotations"] = "Info"
        self._coco_projected_variant: str = COCO133_PREDICTION_LAYER_TO_PATH[Coco133AnnotationLayer.PROJECTED_2D]
        self._coco_gt_class_id: int = int(Coco133AnnotationLayer.GT)

    def initialize_rrd(self, rrd_path, state):
        yield from self._initialize_rrd(rrd_path, state)

    def _initialize_rrd(self, rrd_path: str | None, state: gr.State | AppState):
        if rrd_path is None:
            yield None, state
            return

        resolved_path: Path = Path(rrd_path)

        if not resolved_path.exists():
            raise gr.Error(f"RRD file not found: {resolved_path}")

        try:
            dataset_cfg: RRDExoEgoConfig = RRDExoEgoConfig(rrd_path=resolved_path)
            exoego_sequence: BaseExoEgoSequence = dataset_cfg.setup()
        except Exception as exc:  # noqa: BLE001
            raise gr.Error(f"Failed to load RRD dataset: {exc}") from exc

        self._current_sequence: BaseExoEgoSequence = exoego_sequence
        self._confirmed_keypoints = {}
        self._projected_keypoints_cache = {}
        self._source_rrd_path = resolved_path

        updated_state: AppState = replace(state, recording_id=uuid.uuid4(), confirmed_keypoints={})
        recording: rr.RecordingStream = get_recording(updated_state.recording_id)
        stream: rr.BinaryStream = recording.binary_stream()

        assert isinstance(updated_state, AppState), "initialize_rrd requires an AppState to provide logging paths."
        parent_log_path: Path = updated_state.rr_log_paths.parent_log_path
        timeline: str = updated_state.rr_log_paths.timeline

        rr.log(
            "/",
            exoego_sequence.world_coordinate_system,
            static=True,
            recording=recording,
        )
        set_annotation_context(recording=recording)

        # Setup the scene and log initial data
        scene_setup_result: SceneSetupResult = setup_scene(
            exoego_sequence=exoego_sequence,
            parent_log_path=parent_log_path,
            timeline=timeline,
            recording=recording,
        )
        shortest_timestamp: Int[ndarray, "n_frames"] = scene_setup_result.shortest_timestamp

        exo_video_log_paths: list[Path] = (
            list(scene_setup_result.log_paths.exo_video_log_paths)
            if scene_setup_result.log_paths.exo_video_log_paths is not None
            else []
        )
        ego_video_log_paths: list[Path] = (
            list(scene_setup_result.log_paths.ego_video_log_paths)
            if scene_setup_result.log_paths.ego_video_log_paths is not None
            else []
        )

        ego_rgb_video_path: Path | None = None
        for video_path in ego_video_log_paths:
            parts: tuple[str, ...] = video_path.parts
            if len(parts) >= 5 and parts[1] == "ego" and parts[2] == "rgb":
                ego_rgb_video_path = video_path
                break
        if ego_rgb_video_path is None and ego_video_log_paths:
            ego_rgb_video_path = ego_video_log_paths[0]

        # Log environment mesh and exoego batch data
        with recording:
            log_environment_mesh(
                exoego_sequence=exoego_sequence,
                parent_log_path=parent_log_path,
            )
            log_exoego_batch(
                exoego_sequence=exoego_sequence,
                parent_log_path=parent_log_path,
                timeline=timeline,
                shortest_timestamp=shortest_timestamp,
                log_ego=True,
                log_exo=True,
                log_mano=True,
            )
            ego_pinhole_paths: list[Path] = sorted(
                {video_path.parent for video_path in ego_video_log_paths},
                key=str,
            )
            if ego_pinhole_paths:
                self._overwrite_ego_projected_keypoints(
                    recording=recording,
                    pinhole_paths=ego_pinhole_paths,
                    timeline=timeline,
                    shortest_timestamp=shortest_timestamp,
                )

        exo_video_log_paths_opt: list[Path] | None = exo_video_log_paths or None
        ego_video_log_paths_opt: list[Path] | None = ego_video_log_paths or None

        video_timestamps_map: dict[str, Int[np.ndarray, "n_frames"]] = dict(updated_state.video_timestamps_by_path)
        ego_sequence = self._current_sequence.ego_sequence if self._current_sequence is not None else None
        if ego_sequence is not None and ego_video_log_paths_opt is not None:
            ego_video_files: list[Path] = ego_sequence.ego_video_paths
            for video_log_path, video_file in zip(ego_video_log_paths_opt, ego_video_files, strict=True):
                frame_timestamps_raw: Int[ndarray, "n_frames"] = np.asarray(
                    rr.AssetVideo(path=video_file).read_frame_timestamps_nanos(),
                    dtype=np.int64,
                )
                canonical_key: str = self._canonical_entity_path(video_log_path)
                video_timestamps_map[canonical_key] = frame_timestamps_raw

        new_rr_paths = replace(
            updated_state.rr_log_paths,
            exo_video_log_paths=exo_video_log_paths_opt,
            ego_video_log_paths=ego_video_log_paths_opt,
        )

        updated_state = replace(
            updated_state,
            rr_log_paths=new_rr_paths,
            shortest_timestamps=shortest_timestamp,
            video_timestamps_by_path=video_timestamps_map,
        )

        container: rrb.ContainerLike = create_container(
            exo_video_log_paths=exo_video_log_paths_opt,
            ego_video_log_paths=ego_video_log_paths_opt,
        )
        blueprint = rrb.Blueprint(
            rrb.Horizontal(
                contents=[container],
                column_shares=[4, 1],
            ),
            collapse_panels=True,
        )
        rr.send_blueprint(blueprint, recording=recording)
        stream.flush()
        payload_bytes: bytes | None = stream.read()

        yield payload_bytes, updated_state

    def save_annotated_rrd(self, state: AppState, progress=gr.Progress()):
        if self._current_sequence is None:
            raise gr.Error("No RRD is currently loaded; cannot save annotated RRD.")
        progress(0.0, desc="Preparing to save annotated RRD")
        with tempfile.NamedTemporaryFile(prefix="rrd_label_results_", suffix=".rrd", delete=False) as temp_file:
            # clean up temporary file later, but also probably upload to huggingface dataset?
            # pending_cleanup.append(temp_file.name)
            # update state with the path to save
            updated_state: AppState = replace(state, rrd_save_path=Path(temp_file.name))
            raw_rrd_path: Path = updated_state.rrd_save_path
            rr_config: RerunTyroConfig = RerunTyroConfig(save=updated_state.rrd_save_path)
            exoego_sequence: BaseExoEgoSequence = self._current_sequence
            parent_log_path: Path = updated_state.rr_log_paths.parent_log_path
            timeline: str = updated_state.rr_log_paths.timeline
            rr.log(
                "/",
                exoego_sequence.world_coordinate_system,
            )
            set_annotation_context(recording=None)

            # Setup the scene and log initial data
            scene_setup_result: SceneSetupResult = setup_scene(
                exoego_sequence=exoego_sequence,
                parent_log_path=parent_log_path,
                timeline=timeline,
            )
            shortest_timestamp: Int[ndarray, "n_frames"] = scene_setup_result.shortest_timestamp

            exo_video_log_paths: list[Path] = (
                list(scene_setup_result.log_paths.exo_video_log_paths)
                if scene_setup_result.log_paths.exo_video_log_paths is not None
                else []
            )
            ego_video_log_paths: list[Path] = (
                list(scene_setup_result.log_paths.ego_video_log_paths)
                if scene_setup_result.log_paths.ego_video_log_paths is not None
                else []
            )

            ego_rgb_video_path: Path | None = None
            for video_path in ego_video_log_paths:
                parts: tuple[str, ...] = video_path.parts
                if len(parts) >= 5 and parts[1] == "ego" and parts[2] == "rgb":
                    ego_rgb_video_path = video_path
                    break
            if ego_rgb_video_path is None and ego_video_log_paths:
                ego_rgb_video_path = ego_video_log_paths[0]

            # Log environment mesh and exoego batch data
            log_environment_mesh(
                exoego_sequence=exoego_sequence,
                parent_log_path=parent_log_path,
            )
            log_exoego_batch(
                exoego_sequence=exoego_sequence,
                parent_log_path=parent_log_path,
                timeline=timeline,
                shortest_timestamp=shortest_timestamp,
                log_ego=True,
                log_exo=True,
                log_mano=True,
            )
            ego_pinhole_paths_save: list[Path] = sorted(
                {video_path.parent for video_path in ego_video_log_paths},
                key=str,
            )
            if ego_pinhole_paths_save:
                self._overwrite_ego_projected_keypoints(
                    recording=None,
                    pinhole_paths=ego_pinhole_paths_save,
                    timeline=timeline,
                    shortest_timestamp=shortest_timestamp,
                )

            if updated_state.confirmed_keypoints:
                self._log_confirmed_keypoints_for_save(updated_state.confirmed_keypoints)

            container: rrb.ContainerLike = create_container(
                exo_video_log_paths=exo_video_log_paths,
                ego_video_log_paths=ego_video_log_paths,
            )
            blueprint = rrb.Blueprint(
                rrb.Horizontal(
                    contents=[container],
                    column_shares=[4, 1],
                ),
                collapse_panels=True,
            )
            progress(0.7, desc="Saving viewer blueprint")
            application_id: str = rr_config.application_id
            blueprint_path: Path = raw_rrd_path.with_suffix(".rbl")
            if blueprint_path.exists():
                blueprint_path.unlink()
            blueprint.save(application_id=application_id, path=str(blueprint_path))

            progress(0.85, desc="Compacting annotated recording")
            compacted_rrd_path: Path = raw_rrd_path.with_name(f"{raw_rrd_path.stem}_with_blueprint.rrd")
            compact_cmd: list[str] = [
                "rerun",
                "rrd",
                "compact",
                str(blueprint_path),
                str(raw_rrd_path),
                "-o",
                str(compacted_rrd_path),
            ]
            try:
                run(compact_cmd, check=True, capture_output=True, text=True)
            except CalledProcessError as exc:
                error_message: str = exc.stderr or exc.stdout or str(exc)
                raise gr.Error(f"Failed to compact annotated recording: {error_message}") from exc

            raw_rrd_path.unlink(missing_ok=True)
            blueprint_path.unlink(missing_ok=True)
            updated_state = replace(updated_state, rrd_save_path=compacted_rrd_path)

        progress(1.0, desc="Pipeline complete")
        return updated_state, str(updated_state.rrd_save_path)

    def _overwrite_ego_projected_keypoints(
        self,
        *,
        recording: rr.RecordingStream | None,
        pinhole_paths: list[Path],
        timeline: str,
        shortest_timestamp: Int[ndarray, "n_frames"],
    ) -> None:
        if self._source_rrd_path is None:
            raise gr.Error("Unable to relog projected keypoints; source RRD path is unknown.")

        try:
            archive = rr.dataframe.load_archive(str(self._source_rrd_path))
        except Exception as exc:  # noqa: BLE001
            raise gr.Error(f"Failed to load source RRD '{self._source_rrd_path}': {exc}") from exc

        source_recordings: list[rr.dataframe.Recording] = list(archive.all_recordings())
        if not source_recordings:
            raise gr.Error(f"No recordings found within RRD '{self._source_rrd_path}'.")

        n_keypoints: int = len(COCO_133_IDS)
        for pinhole_path in pinhole_paths:
            projected_entity_path: Path = pinhole_path / "pred" / "coco133_uv" / self._coco_projected_variant
            projected_entity: str = self._with_leading_slash(projected_entity_path)
            table = self._read_entity_table(source_recordings, projected_entity)
            if table is None or table.num_rows == 0:
                raise gr.Error(
                    f"Projected keypoints missing at '{projected_entity}'. "
                    "Ensure the original RRD contains the expected prediction layer."
                )

            positions_stack, confidences_stack = self._extract_uv_arrays(
                table=table,
                entity_path=projected_entity,
                expected_keypoints=n_keypoints,
            )

            n_frames_available: int = int(
                min(
                    positions_stack.shape[0],
                    confidences_stack.shape[0],
                    shortest_timestamp.shape[0],
                )
            )
            if n_frames_available <= 0:
                raise gr.Error(
                    f"Projected keypoints at '{projected_entity}' contain no time-aligned samples."
                )

            positions_trim: Float[ndarray, "n_frames n_kpts 2"] = positions_stack[:n_frames_available]
            confidences_trim: Float[ndarray, "n_frames n_kpts"] = confidences_stack[:n_frames_available]
            colors_trim: UInt8[ndarray, "n_frames n_kpts 3"] = confidence_scores_to_rgb(
                confidences_trim[..., np.newaxis]
            )
            timestamps_trim: Float[ndarray, "n_frames"] = shortest_timestamp[:n_frames_available].astype(np.float64)

            target_entity: str = self._with_leading_slash(pinhole_path / "coco133_uv")
            canonical_target: str = self._canonical_entity_path(target_entity)
            self._projected_keypoints_cache[canonical_target] = (
                positions_trim.astype(np.float32, copy=True),
                confidences_trim.astype(np.float32, copy=True),
            )
            self._log_points2d_with_confidence(
                entity_path=target_entity,
                positions=positions_trim,
                confidences=confidences_trim,
                colors=colors_trim,
                timeline=timeline,
                timestamps_ns=timestamps_trim,
                recording=recording,
            )

    @staticmethod
    def _read_entity_table(
        recordings: list[Recording],
        entity_path: str,
    ):
        for source_recording in recordings:
            index_column: str | None = Controller._preferred_index_column(source_recording)
            if index_column is None:
                continue
            try:
                view = source_recording.view(index=index_column, contents=entity_path)
            except Exception:  # noqa: BLE001
                continue
            try:
                table = view.select().read_all()
            except Exception:  # noqa: BLE001
                continue
            if table is not None and table.num_rows > 0:
                return table
        return None

    @staticmethod
    def _extract_uv_arrays(
        *,
        table,
        entity_path: str,
        expected_keypoints: int,
    ) -> tuple[Float[ndarray, "n_frames n_kpts 2"], Float[ndarray, "n_frames n_kpts"]]:
        positions_column: str = f"{entity_path}:Points2D:positions"
        confidences_column: str = f"{entity_path}:simplecv.KeypointConfidence2D:confidences"

        if positions_column not in table.column_names:
            raise gr.Error(f"Entity '{entity_path}' is missing Points2D positions.")
        if confidences_column not in table.column_names:
            raise gr.Error(f"Entity '{entity_path}' is missing keypoint confidences.")

        positions_entries = table.column(positions_column).combine_chunks().to_pylist()
        confidences_entries = table.column(confidences_column).combine_chunks().to_pylist()

        n_frames: int = table.num_rows
        positions_stack: Float[ndarray, "n_frames n_kpts 2"] = np.full(
            (n_frames, expected_keypoints, 2),
            np.nan,
            dtype=np.float32,
        )
        confidences_stack: Float[ndarray, "n_frames n_kpts"] = np.zeros(
            (n_frames, expected_keypoints),
            dtype=np.float32,
        )

        for frame_idx, (positions_entry, confidences_entry) in enumerate(
            zip(positions_entries, confidences_entries, strict=False)
        ):
            normalized_positions = Controller._normalize_nested_list(positions_entry)
            normalized_confidences = Controller._normalize_nested_list(confidences_entry)
            if normalized_positions is None or normalized_confidences is None:
                continue

            positions_array: Float[ndarray, "n_kpts_frame 2"] = np.asarray(
                normalized_positions,
                dtype=np.float32,
            )
            confidences_array: Float[ndarray, "n_kpts_frame"] = np.asarray(
                normalized_confidences,
                dtype=np.float32,
            )
            n_keypoints_frame: int = int(
                min(
                    positions_array.shape[0],
                    confidences_array.shape[0],
                    expected_keypoints,
                )
            )
            if n_keypoints_frame == 0:
                continue

            positions_stack[frame_idx, :n_keypoints_frame] = positions_array[:n_keypoints_frame]
            confidences_stack[frame_idx, :n_keypoints_frame] = confidences_array[:n_keypoints_frame]

        return positions_stack, confidences_stack

    @staticmethod
    def _normalize_nested_list(values) -> list | None:
        if values is None:
            return None
        result = values
        while isinstance(result, list) and len(result) == 1 and isinstance(result[0], list):
            result = result[0]
        return result if isinstance(result, list) else None

    def _log_points2d_with_confidence(
        self,
        *,
        entity_path: str,
        positions: Float[ndarray, "n_frames n_kpts 2"],
        confidences: Float[ndarray, "n_frames n_kpts"],
        colors: UInt8[ndarray, "n_frames n_kpts 3"],
        timeline: str,
        timestamps_ns: Float[ndarray, "n_frames"],
        recording: rr.RecordingStream | None,
    ) -> None:
        n_frames: int = positions.shape[0]
        keypoint_lengths: Int[ndarray, "n_frames"] = np.full(n_frames, positions.shape[1], dtype=np.int32)
        positions_flat: Float[ndarray, "n_total 2"] = positions.reshape(-1, 2).astype(np.float32, copy=False)
        confidences_flat: Float[ndarray, "n_total"] = confidences.reshape(-1).astype(np.float32, copy=False)
        colors_flat: UInt8[ndarray, "n_total 3"] = colors.reshape(-1, 3).astype(np.uint8, copy=False)
        durations: Float[ndarray, "n_frames"] = 1e-9 * timestamps_ns

        if recording is not None:
            rr.log(entity_path, rr.Clear(recursive=True), recording=recording)
            rr.log(
                entity_path,
                Points2DWithConfidence.from_fields(
                    class_ids=self._coco_gt_class_id,
                    keypoint_ids=COCO_133_IDS,
                    show_labels=False,
                ),
                static=True,
                recording=recording,
            )
            rr.send_columns(
                entity_path,
                indexes=[rr.TimeColumn(timeline, duration=durations)],
                columns=[
                    *Points2DWithConfidence.columns(
                        positions=positions_flat,
                        confidences=confidences_flat,
                        colors=colors_flat,
                    ).partition(keypoint_lengths),
                ],
                recording=recording,
            )
            return

        rr.log(entity_path, rr.Clear(recursive=True))
        rr.log(
            entity_path,
            Points2DWithConfidence.from_fields(
                class_ids=self._coco_gt_class_id,
                keypoint_ids=COCO_133_IDS,
                show_labels=False,
            ),
            static=True,
        )
        rr.send_columns(
            entity_path,
            indexes=[rr.TimeColumn(timeline, duration=durations)],
            columns=[
                *Points2DWithConfidence.columns(
                    positions=positions_flat,
                    confidences=confidences_flat,
                    colors=colors_flat,
                ).partition(keypoint_lengths),
            ],
        )

    @staticmethod
    def _preferred_index_column(recording: Recording) -> str | None:
        index_names: list[str] = [index_column.name for index_column in recording.schema().index_columns()]
        for candidate in ("video_time", "log_time", "log_tick", "frame_nr"):
            if candidate in index_names:
                return candidate
        return index_names[0] if index_names else None

    @staticmethod
    def _with_leading_slash(path: Path | str) -> str:
        path_str: str = str(path)
        if not path_str.startswith("/"):
            path_str = "/" + path_str.lstrip("/")
        return path_str.rstrip("/")

    @staticmethod
    def _log_confirmed_keypoints_for_save(
        confirmed: dict[str, dict[int, ConfirmedKeypointRecord]],
    ) -> None:
        """Re-log cached keypoint annotations into the persisted RRD."""
        for path_records in confirmed.values():
            for timestamp_ns in sorted(path_records):
                record: ConfirmedKeypointRecord = path_records[timestamp_ns]
                rr.set_time(record.timeline, duration=record.timestamp_ns * 1e-9)
                rr.log(
                    str(record.entity_path),
                    Points2DWithConfidence(
                        positions=record.positions,
                        confidences=record.confidences,
                        class_ids=record.class_id,
                        keypoint_ids=list(record.keypoint_ids),
                        show_labels=False,
                        colors=record.colors,
                    ),
                )

    def log_bbox_kpts(self, state, corner_selector, hand):
        yield from self._log_bbox_kpts(state, corner_selector, hand)

    def _log_bbox_kpts(
        self,
        state: AppState,
        corner_selector: Literal["Top Left", "Bottom Right"],
        hand: Literal["Left Hand", "Right Hand"],
    ):
        recording: rr.RecordingStream = get_recording(state.recording_id)
        stream: rr.BinaryStream = recording.binary_stream()
        evt: SelectionChangeEvent | None = state.selection_evt

        if evt is None:
            yield stream.read(), state, corner_selector
            return

        item: EntitySelectionItem = evt.items[0]  # type: ignore[index]
        item_path: Path = Path(item.entity_path)
        video_entity_path: Path = self._derive_video_entity_path(item_path)
        uvz: list[float] | None = item.position

        # Clear the selection event after processing
        cleared_state: AppState = replace(state, selection_evt=None)
        next_corner_label: Literal["Top Left", "Bottom Right"] = corner_selector
        updated_state: AppState = replace(cleared_state, active_video_entity_path=video_entity_path)

        # update the bounding box draft in state with the new corner
        bbox_draft: BoundingBoxDraft = state.bounding_box_draft
        current_time_ns: int = state.current_time_ns
        annot_path: Path = video_entity_path.parent / "annot"
        # initialize ts_nano if not set
        if bbox_draft.ts_nano is None:
            bbox_draft = replace(bbox_draft, ts_nano=current_time_ns)

        if bbox_draft.ts_nano is not None and bbox_draft.ts_nano != current_time_ns:
            # TODO clear out the previously logged points for this annotation if the time has changed
            # this might actually make more sense to do in a follow up to the time change callback

            # rr.set_time(state.rr_log_paths.timeline, duration=bbox_draft.ts_nano * 1e-9)
            # rr.log(f"{annot_path}", rr.Clear(recursive=True))

            # User advanced the timeline; discard the prior frame's draft.
            bbox_draft = BoundingBoxDraft()
            bbox_draft = replace(bbox_draft, ts_nano=current_time_ns)

        if uvz is not None:
            uv_coords: Float[ndarray, "2"] = np.array(uvz[0:2], dtype=np.float32)
            has_opposite_corner: bool = False
            bbox_draft = replace(bbox_draft, entity_path=video_entity_path, xyxy=None)
            match corner_selector:
                case "Top Left":
                    has_opposite_corner = bbox_draft.bottom_right is not None
                    bbox_draft = replace(bbox_draft, top_left=uv_coords)
                case "Bottom Right":
                    has_opposite_corner = bbox_draft.top_left is not None
                    bbox_draft = replace(bbox_draft, bottom_right=uv_coords)

            timeline: str = state.rr_log_paths.timeline
            kpt_name: Literal["tl", "br"] = "tl" if corner_selector == "Top Left" else "br"
            annot_kpt_log_path: Path = annot_path / kpt_name
            annot_bbox_log_path: Path = annot_path / "bbox"
            # Toggle the active corner so the next click captures the opposite corner.
            next_corner_label = "Bottom Right" if corner_selector == "Top Left" else "Top Left"

            point_color: list[int] = [0, 255, 0] if corner_selector == "Top Left" else [255, 0, 0]
            point_label: str = f"{hand.split()[0]}-{corner_selector.replace(' ', '').lower()}"
            bbox_xyxy: Float[ndarray, "1 4"] | None = None

            # If both corners are available, prepare the bounding box data for logging.
            if has_opposite_corner and bbox_draft.top_left is not None and bbox_draft.bottom_right is not None:
                top_left_coords: Float[ndarray, "2"] = bbox_draft.top_left
                bottom_right_coords: Float[ndarray, "2"] = bbox_draft.bottom_right
                mins: Float[ndarray, "2"] = np.minimum(top_left_coords, bottom_right_coords)
                maxs: Float[ndarray, "2"] = np.maximum(top_left_coords, bottom_right_coords)
                bbox_xyxy = np.array(
                    [[mins[0], mins[1], maxs[0], maxs[1]]],
                    dtype=np.float32,
                )
                bbox_draft = replace(bbox_draft, xyxy=bbox_xyxy)

            with recording:
                rr.set_time(timeline, duration=current_time_ns * 1e-9)
                rr.log(
                    f"{annot_kpt_log_path}",
                    rr.Points2D(
                        positions=uv_coords,
                        colors=[point_color],
                        labels=[point_label],
                        radii=[6.0],
                    ),
                )
                if bbox_xyxy is not None:
                    rr.log(
                        f"{annot_bbox_log_path}",
                        rr.Boxes2D(
                            array=bbox_xyxy,
                            array_format=rr.Box2DFormat.XYXY,
                            labels=[f"{hand}"],
                        ),
                    )

            updated_state = replace(updated_state, bounding_box_draft=bbox_draft)

        payload: bytes | None = stream.read()
        yield payload, updated_state, next_corner_label

    def confirm_bbox(self, state, hand_choice):
        yield from self._confirm_bbox(state, hand_choice)

    def _confirm_bbox(self, state: AppState, hand_choice: Literal["Left Hand", "Right Hand"]):
        if not isinstance(state, AppState):
            raise gr.Error("Confirming a bounding box requires application state.")

        recording: rr.RecordingStream = get_recording(state.recording_id)
        stream: rr.BinaryStream = recording.binary_stream()

        if self._current_sequence is None or self._current_sequence.ego_sequence is None:
            raise gr.Error("No ego videos are loaded; please initialize an RRD recording before confirming.")

        bbox_draft: BoundingBoxDraft = state.bounding_box_draft
        top_left_opt: Float[ndarray, "2"] | None = bbox_draft.top_left
        bottom_right_opt: Float[ndarray, "2"] | None = bbox_draft.bottom_right
        if top_left_opt is None or bottom_right_opt is None:
            raise gr.Error("Please label both top-left and bottom-right corners before confirming the bounding box.")

        if bbox_draft.ts_nano is None:
            raise gr.Error("The bounding box draft has no timestamp; reselect the frame and try again.")

        current_time_ns: int = state.current_time_ns
        if bbox_draft.ts_nano != current_time_ns:
            raise gr.Error(
                "The bounding box draft was created at a different timestamp. "
                "Scrub back to the original frame or re-label both corners."
            )

        entity_path: Path | None = bbox_draft.entity_path
        if entity_path is None:
            raise gr.Error("Missing the source video stream for this annotation. Please click the video frame again.")

        video_entity_path: Path = entity_path

        canonical_video_path: str = self._canonical_entity_path(video_entity_path)
        video_timestamps_opt: Int[np.ndarray, "n_frames"] | None = state.video_timestamps_by_path.get(
            canonical_video_path
        )
        if video_timestamps_opt is None:
            raise gr.Error(
                f"No timestamp metadata found for '{canonical_video_path}'. Reload the dataset and try again."
            )

        frame_idx: int = self._frame_index_from_timestamps(video_timestamps_opt, current_time_ns)
        ego_sequence = self._current_sequence.ego_sequence
        ego_sequence = self._current_sequence.ego_sequence
        assert ego_sequence is not None
        camera_name: str = self._extract_camera_name(video_entity_path)
        try:
            camera_idx: int = ego_sequence.ego_video_names.index(camera_name)
        except ValueError as exc:
            raise gr.Error(
                f"Camera '{camera_name}' is not part of the loaded ego sequence; select a valid ego stream."
            ) from exc

        bgr_frame_list = ego_sequence.ego_video_readers[frame_idx]
        if camera_idx >= len(bgr_frame_list):
            raise gr.Error(
                f"Frame {frame_idx} does not include camera '{camera_name}'. Please reload the recording and retry."
            )

        bgr_frame = bgr_frame_list[camera_idx]
        if bgr_frame is None:
            raise gr.Error(
                f"Missing ego frame {frame_idx} for camera '{camera_name}'. Reselect the frame and try again."
            )

        bgr_hw3: UInt8[ndarray, "H W 3"] = np.asarray(bgr_frame, dtype=np.uint8)
        if bgr_hw3.ndim != 3 or bgr_hw3.shape[2] != 3:
            raise gr.Error(f"Expected an HxWx3 frame for camera '{camera_name}', received shape {bgr_hw3.shape}.")

        rgb_hw3: UInt8[ndarray, "H W 3"] = bgr_hw3[..., ::-1].copy()

        top_left: Float[ndarray, "2"] = top_left_opt
        bottom_right: Float[ndarray, "2"] = bottom_right_opt
        bbox_xyxy_opt: Float[ndarray, "1 4"] | None = bbox_draft.xyxy
        if bbox_xyxy_opt is None:
            mins: Float[ndarray, "2"] = np.minimum(top_left, bottom_right)
            maxs: Float[ndarray, "2"] = np.maximum(top_left, bottom_right)
            bbox_xyxy: Float[ndarray, "1 4"] = np.array(
                [[mins[0], mins[1], maxs[0], maxs[1]]],
                dtype=np.float32,
            )
        else:
            bbox_xyxy = bbox_xyxy_opt.astype(np.float32, copy=True)

        hand_slug: str
        handedness: Literal["left", "right"]
        hand_slug, handedness = self._hand_choice_to_metadata(hand_choice)

        wilor_pred = self.engine.infer_hand_keypoints(
            rgb_hw3=rgb_hw3,
            xyxy=bbox_xyxy,
            handedness=handedness,
        )

        keypoints_uv: Float[ndarray, "n_kpts=21 2"] = wilor_pred.pred_keypoints_2d[0].astype(np.float32, copy=True)
        confidences: Float[ndarray, "n_kpts=21"] = wilor_pred.confidence_2d[0].astype(np.float32, copy=True)
        hand_indices: np.ndarray
        match handedness:
            case "left":
                hand_indices = LEFT_HAND_IDX
            case "right":
                hand_indices = RIGHT_HAND_IDX
            case _:
                raise gr.Error(f"Unsupported handedness '{handedness}'.")

        pinhole_root: Path = video_entity_path.parent  # .../<cam>/pinhole
        pred_path: Path = pinhole_root / "coco133_uv"
        canonical_pred_path: str = self._canonical_entity_path(pred_path)

        cache_key: tuple[str, int] = (canonical_video_path, frame_idx)
        cached_pair: tuple[Float[ndarray, "n_kpts 2"], Float[ndarray, "n_kpts"]] | None = self._confirmed_keypoints.get(
            cache_key
        )
        base_positions: Float[ndarray, "n_kpts 2"]
        base_confidences: Float[ndarray, "n_kpts"]
        if cached_pair is not None:
            cached_positions, cached_confidences = cached_pair
            base_positions = cached_positions.astype(np.float32, copy=True)
            base_confidences = cached_confidences.astype(np.float32, copy=True)
        else:
            cached_sequence: tuple[
                Float[ndarray, "n_frames n_kpts 2"],
                Float[ndarray, "n_frames n_kpts"],
            ] | None = self._projected_keypoints_cache.get(canonical_pred_path)
            if cached_sequence is not None and frame_idx < cached_sequence[0].shape[0]:
                positions_by_frame, confidences_by_frame = cached_sequence
                base_positions = positions_by_frame[frame_idx].astype(np.float32, copy=True)
                base_confidences = confidences_by_frame[frame_idx].astype(np.float32, copy=True)
            else:
                base_positions_raw, base_confidences_raw = self._ego_uv_for_frame(camera_name, frame_idx)
                if base_positions_raw is None or base_confidences_raw is None:
                    base_positions = np.full(
                        (len(COCO_133_IDS), 2),
                        np.nan,
                        dtype=np.float32,
                    )
                    base_confidences = np.full(
                        len(COCO_133_IDS),
                        np.nan,
                        dtype=np.float32,
                    )
                else:
                    base_positions = base_positions_raw.astype(np.float32, copy=True)
                    base_confidences = base_confidences_raw.astype(np.float32, copy=True)

        positions_full: Float[ndarray, "n_kpts 2"] = base_positions.astype(np.float32, copy=False)
        confidences_full: Float[ndarray, "n_kpts"] = base_confidences.astype(np.float32, copy=False)
        positions_full[hand_indices, :] = keypoints_uv
        confidences_full[hand_indices] = confidences
        confidence_rgb_full_stack: UInt8[ndarray, "1 n_kpts 3"] = confidence_scores_to_rgb(
            confidences_full[np.newaxis, :, np.newaxis]
        )
        confidence_rgb_full: UInt8[ndarray, "n_kpts 3"] = confidence_rgb_full_stack[0]

        annot_root_path: Path = video_entity_path.parent / "annot"
        timeline: str = state.rr_log_paths.timeline

        with recording:
            rr.set_time(timeline, duration=current_time_ns * 1e-9)
            rr.log(
                f"{pred_path}",
                Points2DWithConfidence(
                    positions=positions_full,
                    confidences=confidences_full,
                    class_ids=self._coco_gt_class_id,
                    keypoint_ids=COCO_133_IDS,
                    show_labels=False,
                    colors=confidence_rgb_full,
                ),
            )
            rr.log(f"{annot_root_path}", rr.Clear(recursive=True))

        positions_snapshot: Float[ndarray, "n_kpts 2"] = positions_full.astype(np.float32, copy=True)
        confidences_snapshot: Float[ndarray, "n_kpts"] = confidences_full.astype(np.float32, copy=True)
        colors_snapshot: UInt8[ndarray, "n_kpts 3"] = confidence_rgb_full.astype(np.uint8, copy=True)
        self._confirmed_keypoints[cache_key] = (
            positions_snapshot,
            confidences_snapshot,
        )
        cache_entry = self._projected_keypoints_cache.get(canonical_pred_path)
        if cache_entry is not None:
            positions_by_frame, confidences_by_frame = cache_entry
            if frame_idx < positions_by_frame.shape[0]:
                positions_by_frame[frame_idx] = positions_snapshot.astype(np.float32, copy=True)
                confidences_by_frame[frame_idx] = confidences_snapshot.astype(np.float32, copy=True)
            else:
                new_length: int = frame_idx + 1
                n_kpts: int = positions_snapshot.shape[0]
                extended_positions: Float[ndarray, "n_frames n_kpts 2"] = np.full(
                    (new_length, n_kpts, 2),
                    np.nan,
                    dtype=np.float32,
                )
                extended_confidences: Float[ndarray, "n_frames n_kpts"] = np.zeros(
                    (new_length, n_kpts),
                    dtype=np.float32,
                )
                extended_positions[: positions_by_frame.shape[0]] = positions_by_frame
                extended_confidences[: confidences_by_frame.shape[0]] = confidences_by_frame
                extended_positions[frame_idx] = positions_snapshot.astype(np.float32, copy=True)
                extended_confidences[frame_idx] = confidences_snapshot.astype(np.float32, copy=True)
                self._projected_keypoints_cache[canonical_pred_path] = (
                    extended_positions,
                    extended_confidences,
                )
        else:
            self._projected_keypoints_cache[canonical_pred_path] = (
                positions_snapshot[np.newaxis, ...].astype(np.float32, copy=True),
                confidences_snapshot[np.newaxis, ...].astype(np.float32, copy=True),
            )
        new_confirmed_by_path: dict[str, dict[int, ConfirmedKeypointRecord]] = dict(state.confirmed_keypoints)
        path_records: dict[int, ConfirmedKeypointRecord] = dict(
            new_confirmed_by_path.get(canonical_video_path, {})
        )
        confirmed_record = ConfirmedKeypointRecord(
            entity_path=pred_path,
            timeline=timeline,
            timestamp_ns=current_time_ns,
            positions=positions_snapshot,
            confidences=confidences_snapshot,
            colors=colors_snapshot,
            class_id=self._coco_gt_class_id,
            keypoint_ids=tuple(int(idx) for idx in COCO_133_IDS),
        )
        path_records[current_time_ns] = confirmed_record
        new_confirmed_by_path[canonical_video_path] = path_records

        updated_state: AppState = replace(
            state,
            bounding_box_draft=BoundingBoxDraft(),
            confirmed_keypoints=new_confirmed_by_path,
        )

        payload: bytes | None = stream.read()
        yield payload, updated_state, "Top Left"

    def clear_current_keypoints(self, state):
        yield from self._clear_current_keypoints(state)

    def _clear_current_keypoints(self, state: AppState):
        if not isinstance(state, AppState):
            raise gr.Error("Clearing keypoints requires application state.")

        recording: rr.RecordingStream = get_recording(state.recording_id)
        stream: rr.BinaryStream = recording.binary_stream()

        if self._current_sequence is None or self._current_sequence.ego_sequence is None:
            raise gr.Error("No ego videos are loaded; initialize an RRD recording before clearing.")

        bbox_draft: BoundingBoxDraft = state.bounding_box_draft
        entity_path: Path | None = state.active_video_entity_path or bbox_draft.entity_path
        if entity_path is None:
            raise gr.Error("Click an ego RGB frame before clearing keypoints.")

        canonical_video_path: str = self._canonical_entity_path(entity_path)
        video_timestamps_opt: Int[np.ndarray, "n_frames"] | None = state.video_timestamps_by_path.get(
            canonical_video_path
        )
        if video_timestamps_opt is None:
            raise gr.Error(
                f"No timestamp metadata found for '{canonical_video_path}'. Reload the dataset and try again."
            )

        frame_idx: int = self._frame_index_from_timestamps(video_timestamps_opt, state.current_time_ns)
        timeline: str = state.rr_log_paths.timeline
        pinhole_root: Path = entity_path.parent
        pred_path: Path = pinhole_root / "coco133_uv"
        canonical_pred_path: str = self._canonical_entity_path(pred_path)

        cache_key: tuple[str, int] = (canonical_video_path, frame_idx)
        self._confirmed_keypoints.pop(cache_key, None)
        cache_entry = self._projected_keypoints_cache.get(canonical_pred_path)
        if cache_entry is not None and frame_idx < cache_entry[0].shape[0]:
            positions_by_frame, confidences_by_frame = cache_entry
            positions_by_frame[frame_idx] = np.full(
                positions_by_frame[frame_idx].shape,
                np.nan,
                dtype=np.float32,
            )
            confidences_by_frame[frame_idx] = np.zeros(
                confidences_by_frame[frame_idx].shape,
                dtype=np.float32,
            )
        updated_confirmed: dict[str, dict[int, ConfirmedKeypointRecord]] = state.confirmed_keypoints
        existing_records: dict[int, ConfirmedKeypointRecord] | None = state.confirmed_keypoints.get(
            canonical_video_path
        )
        if existing_records is not None and state.current_time_ns in existing_records:
            new_confirmed_by_path: dict[str, dict[int, ConfirmedKeypointRecord]] = dict(
                state.confirmed_keypoints
            )
            trimmed_records: dict[int, ConfirmedKeypointRecord] = dict(existing_records)
            trimmed_records.pop(state.current_time_ns, None)
            if trimmed_records:
                new_confirmed_by_path[canonical_video_path] = trimmed_records
            else:
                new_confirmed_by_path.pop(canonical_video_path, None)
            updated_confirmed = new_confirmed_by_path

        with recording:
            rr.set_time(timeline, duration=state.current_time_ns * 1e-9)
            rr.log(f"{pred_path}", rr.Clear(recursive=True))
            rr.log(f"{pinhole_root / 'annot'}", rr.Clear(recursive=True))

        cleared_state: AppState = replace(
            state,
            bounding_box_draft=BoundingBoxDraft(),
            confirmed_keypoints=updated_confirmed,
        )
        payload: bytes | None = stream.read()
        yield payload, cleared_state

    def _ego_uv_for_frame(
        self,
        camera_name: str,
        frame_idx: int,
    ) -> tuple[Float[ndarray, "n_kpts 2"] | None, Float[ndarray, "n_kpts"] | None]:
        if self._current_sequence is None or self._current_sequence.ego_sequence is None:
            return None, None

        exoego_labels = getattr(self._current_sequence, "exoego_labels", None)
        if exoego_labels is None:
            return None, None

        xyzc_stack: Float[ndarray, "n_frames 133 4"] = exoego_labels.xyzc_stack
        if frame_idx >= len(xyzc_stack):
            return None, None

        xyzc_frame: Float[ndarray, "133 4"] = xyzc_stack[frame_idx]
        xyz: Float[ndarray, "133 3"] = xyzc_frame[:, :3]
        conf: Float[ndarray, "133"] = xyzc_frame[:, 3].astype(np.float32, copy=True)
        xyz_hom: Float[ndarray, "133 4"] = np.concatenate(
            [xyz.astype(np.float32, copy=False), np.ones((xyz.shape[0], 1), dtype=np.float32)],
            axis=1,
        )

        ego_cam_dict = self._current_sequence.ego_sequence.ego_cam_dict
        cam_param_list: list[PinholeParameters] | None = ego_cam_dict.get(camera_name)
        if not cam_param_list:
            return None, None

        cam_idx: int = min(frame_idx, len(cam_param_list) - 1)
        pinhole = cam_param_list[cam_idx]
        proj_matrix: Float[ndarray, "3 4"] = pinhole.projection_matrix.astype(np.float32, copy=False)

        proj: Float[ndarray, "133 3"] = xyz_hom @ proj_matrix.T
        w: Float[ndarray, "133"] = proj[:, 2]
        with np.errstate(divide="ignore", invalid="ignore"):
            uv: Float[ndarray, "133 2"] = proj[:, :2] / w[:, None]

        invalid_mask: np.ndarray = (~np.isfinite(uv).all(axis=1)) | (w <= 0) | np.isclose(w, 0.0)
        uv[invalid_mask] = np.nan
        conf[invalid_mask] = 0.0

        uv = filter_out_of_bounds_keypoints(uv[np.newaxis, :, :], pinhole, margin_percentage=0.0)[0]
        out_of_bounds_mask: np.ndarray = np.isnan(uv).any(axis=1)
        conf[out_of_bounds_mask] = 0.0

        return uv.astype(np.float32, copy=False), conf.astype(np.float32, copy=False)

    def _derive_video_entity_path(self, item_path: Path) -> Path:
        parts: list[str] = [part for part in item_path.parts if part not in {"", "/"}]
        if "video" in parts:
            video_idx: int = parts.index("video")
            target_parts: list[str] = parts[: video_idx + 1]
        else:
            try:
                pinhole_idx: int = parts.index("pinhole")
            except ValueError as exc:
                raise gr.Error(
                    "Selected entity is not under a pinhole/video hierarchy. "
                    "Please select an ego video stream before annotating."
                ) from exc
            target_parts = parts[: pinhole_idx + 1] + ["video"]

        return Path("/").joinpath(*target_parts)

    @staticmethod
    def _canonical_entity_path(path: Path | str) -> str:
        as_str: str = str(path)
        canonical: str = as_str.lstrip("/")
        return canonical.rstrip("/")

    @staticmethod
    def _frame_index_from_timestamps(
        timestamps: Int[ndarray, "n_frames"],
        target_ns: int,
    ) -> int:
        timestamp_array: Int[ndarray, "n_frames"] = np.asarray(timestamps, dtype=np.int64)
        if timestamp_array.size == 0:
            raise gr.Error("No frame timestamps available for the selected video stream.")

        latest_idx: int = int(np.searchsorted(timestamp_array, target_ns, side="right")) - 1
        if latest_idx < 0:
            return 0
        if latest_idx >= timestamp_array.size:
            return int(timestamp_array.size - 1)
        return latest_idx

    @staticmethod
    def _extract_camera_name(entity_path: Path) -> str:
        parts: tuple[str, ...] = entity_path.parts
        filtered_parts: list[str] = [part for part in parts if part not in {"", "/"}]
        try:
            ego_index: int = filtered_parts.index("ego")
            return filtered_parts[ego_index + 1]
        except (ValueError, IndexError) as exc:
            raise gr.Error(
                "The selected annotation is not nested under an ego camera. "
                "Please select an ego RGB view before confirming."
            ) from exc

    @staticmethod
    def _hand_choice_to_metadata(hand_choice: str) -> tuple[str, Literal["left", "right"]]:
        match hand_choice:
            case "Left Hand":
                return "left_hand", "left"
            case "Right Hand":
                return "right_hand", "right"
            case _:
                raise gr.Error(f"Unsupported hand selection '{hand_choice}'.")
