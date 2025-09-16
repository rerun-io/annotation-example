# label_app/controller.py
import uuid
from dataclasses import replace
from pathlib import Path
from typing import Literal, assert_never

import cv2
import gradio as gr
import numpy as np
import open3d as o3d
import rerun as rr
import rerun.blueprint as rrb
from jaxtyping import Float, Int, UInt8
from natsort import natsorted
from numpy import ndarray
from simplecv.data.skeleton.mediapipe import MEDIAPIPE_ID2NAME, MEDIAPIPE_LINKS
from simplecv.rerun_log_utils import log_video, log_pinhole
from simplecv.video_io import MultiVideoReader

from annotation_example.dbt_ui.engine import Engine, MVCalibResults
from annotation_example.dbt_ui.recording_utils import get_recording
from annotation_example.dbt_ui.state import AppState, CurrentPrediction


def set_annotation_context(recording: rr.RecordingStream) -> None:
    rr.log(
        "/",
        rr.AnnotationContext(
            [
                rr.ClassDescription(
                    info=rr.AnnotationInfo(id=0, label="Left Hand", color=(0, 0, 255)),
                    keypoint_annotations=[
                        rr.AnnotationInfo(id=id, label=name) for id, name in MEDIAPIPE_ID2NAME.items()
                    ],
                    keypoint_connections=MEDIAPIPE_LINKS,
                ),
                rr.ClassDescription(
                    info=rr.AnnotationInfo(id=1, label="Right Hand", color=(0, 255, 0)),
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
    annotation_tab = rrb.Spatial3DView()
    if state.rr_log_paths.ego_video_log_paths is not None:
        ego_2d_views: rrb.Vertical = rrb.Vertical(
            contents=[
                rrb.Spatial2DView(origin=video_log_path) for video_log_path in state.rr_log_paths.ego_video_log_paths
            ]
        )
        annotation_tab = rrb.Horizontal(
            contents=[annotation_tab, ego_2d_views], name="Annotations", column_shares=[3, 1]
        )

    if state.rr_log_paths.exo_video_log_paths is not None:
        exo_2d_views: rrb.Horizontal = rrb.Horizontal(
            contents=[
                rrb.Spatial2DView(origin=video_log_path) for video_log_path in state.rr_log_paths.exo_video_log_paths
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

        log_path_update: dict[str, list[Path]] = {f"{group}_video_log_paths": video_log_paths}
        state: AppState = replace(state, rr_log_paths=replace(state.rr_log_paths, **log_path_update))

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
