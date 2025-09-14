# label_app/controller.py
import uuid
from dataclasses import replace
from pathlib import Path
from typing import Literal, assert_never

import gradio as gr
import numpy as np
import open3d as o3d
import rerun as rr
import rerun.blueprint as rrb
from jaxtyping import Bool, Float32, Int, UInt8
from natsort import natsorted
from numpy import ndarray
from simplecv.rerun_log_utils import log_video
from simplecv.video_io import MultiVideoReader

from annotation_example.dbt_ui.engine import Engine, MVCalibResults
from annotation_example.dbt_ui.recording_utils import get_recording
from annotation_example.dbt_ui.state import AppState, CurrentPrediction

from wilor_nano.api.wilor_inference import set_annotation_context


def get_recording(
    recording_id: uuid.UUID,
    application_id: str = "Detection By Tracking Annotation",
    rrd_path: Path | None = None,
) -> rr.RecordingStream:
    recording = rr.RecordingStream(application_id=application_id, recording_id=recording_id)
    if rrd_path is not None:
        rr.set_sinks([rr.FileSink(path=rrd_path)], recording=recording)
    return recording


def create_dbt_blueprint(
    recording: rr.RecordingStream, state: AppState, active_tab: Literal["Info", "Annotations"] = "Info"
) -> rrb.Blueprint:
    # Create a DBT blueprint for the given recording and app state.
    info_tab = rrb.TextDocumentView(name="Info")
    # Use a container for the Annotations tab so we can arrange multiple views.
    # NOTE: Tabs.active_tab only matches by name for View children, not Containers.
    #       When the child is a Container, we must select the tab by index instead.

    annotation_tab = rrb.Spatial3DView()
    if state.rr_log_paths.video_log_paths is not None:
        spatial_2d_views: rrb.Horizontal = rrb.Horizontal(
            contents=[rrb.Spatial2DView(origin=video_log_path) for video_log_path in state.rr_log_paths.video_log_paths]
        )
        annotation_tab = rrb.Vertical(
            contents=[annotation_tab, spatial_2d_views], name="Annotations", row_shares=[3, 1]
        )

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

    def _initialize_rrd(self, zip_path: str | None, state: gr.State | AppState, progress=gr.Progress()):
        # switch to Annotations tab on video upload
        self.tab_name = "Annotations"
        set_annotation_context()
        if zip_path is None:
            # create a new recording id to clear out any previous video state
            state: AppState = replace(state, recording_id=uuid.uuid4(), video_paths_list=None)
            yield None, state
            return
        zip_path = Path(zip_path)
        videos_dir: Path = _extract_zip_to_videos_dir(zip_path)
        assert videos_dir.exists(), f"Video path {videos_dir} does not exist!"
        # make sure that we have videos in the directory
        video_path_list: list[Path] = natsorted(videos_dir.glob("*.mp4"))
        assert len(video_path_list) > 0, "No videos found in uploaded zip"
        recording: rr.RecordingStream = get_recording(state.recording_id)

        progress(0, desc="Starting...")

        exo_timestamps: list[Int[ndarray, "num_frames"]] = []
        video_log_paths: list[Path] = []
        video_paths_list: list[Path] = []
        for i, video_path in enumerate(video_path_list):
            video_log_path: Path = state.rr_log_paths.parent_log_path / f"camera_{i}" / "pinhole" / "video"
            frame_timestamps_ns: Int[ndarray, "num_frames"] = log_video(
                video_path=video_path,
                video_log_path=video_log_path,
                timeline=state.rr_log_paths.timeline,
                recording=recording,
            )
            video_log_paths.append(video_log_path)
            video_paths_list.append(video_path)
            exo_timestamps.append(frame_timestamps_ns)

        shortest_timestamp: Int[ndarray, "n_frames"] = min(exo_timestamps, key=len)

        mv_reader = MultiVideoReader(video_path_list)
        self.engine.mv_reader = mv_reader  # cache for later use
        ts_idx: int = 0
        bgr_list: list[UInt8[ndarray, "H W 3"]] = mv_reader[ts_idx]
        rgb_list: list[UInt8[ndarray, "H W 3"]] = [bgr[..., ::-1] for bgr in bgr_list]
        mv_result: MVCalibResults = self.engine.calibrate_mv(state, rgb_list)
        pcd_ds: o3d.geometry.PointCloud = mv_result.pcd
        # log the pointcloud
        filtered_points: Float32[ndarray, "final_points 3"] = np.asarray(pcd_ds.points, dtype=np.float32)
        filtered_colors: Float32[ndarray, "final_points 3"] = np.asarray(pcd_ds.colors, dtype=np.float32)

        rr.log(
            f"{state.rr_log_paths.parent_log_path}/point_cloud",
            rr.Points3D(
                filtered_points,
                colors=filtered_colors,
            ),
            static=True,
            recording=recording,
        )

        # update rr_paths in state
        rr_log_paths = state.rr_log_paths
        rr_log_paths.video_log_paths = video_log_paths
        state = replace(state, rr_log_paths=rr_log_paths)

        blueprint: rrb.Blueprint = create_dbt_blueprint(recording, state, active_tab=self.tab_name)
        rr.send_blueprint(blueprint, recording=recording)

        # # Update state immutably with video path & frame timestamps
        new_state: AppState = replace(
            state,
            video_paths_list=video_paths_list,
            frame_timestamps_ns=shortest_timestamp,
        )
        yield recording.binary_stream().read(), new_state

    # def on_nav(self, state: AppState, kind: Action) -> AppState:
    #     """Handle navigation events."""
    #     new_state: AppState = reduce(state, kind)
    #     return new_state
