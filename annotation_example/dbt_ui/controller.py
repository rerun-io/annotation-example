# label_app/controller.py
import uuid
from dataclasses import replace
from pathlib import Path
from typing import Literal, assert_never

import gradio as gr
import rerun as rr
import rerun.blueprint as rrb
from jaxtyping import Int
from numpy import ndarray
from simplecv.rerun_log_utils import log_video

from annotation_example.dbt_ui.engine import Engine
from annotation_example.dbt_ui.state import AppState, CurrentPrediction


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
    annotation_tab = rrb.Horizontal(
        contents=[
            rrb.Spatial2DView(),
            rrb.Spatial2DView(),
        ],
        name="Annotations",
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


class Controller:
    """Glue between UI events, Engine calls, pure state transitions, and Rerun logging."""

    def __init__(
        self,
        app_name: str = "label_app",
        engine: Engine | None = None,
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

    def log_video_upload(self, video, state):
        yield from self._log_video_upload(video, state)

    def _log_video_upload(self, video: str | None, state: gr.State | AppState):
        if video is None:
            # create a new recording id to clear out any previous video state
            state: AppState = replace(state, recording_id=uuid.uuid4(), video_path=None)
            yield None, state
            return
        video_path = Path(video)
        assert video_path.exists(), f"Video path {video_path} does not exist!"
        recording: rr.RecordingStream = get_recording(state.recording_id)
        # switch to Annotations tab on video upload
        self.tab_name = "Annotations"
        blueprint: rrb.Blueprint = create_dbt_blueprint(recording, state, active_tab=self.tab_name)
        rr.send_blueprint(blueprint, recording=recording)
        frame_timestamps_ns: Int[ndarray, "num_frames"] = log_video(
            video_path=video_path,
            video_log_path=state.rr_log_paths.parent_log_path / "video",
            timeline=state.rr_log_paths.timeline,
            recording=recording,
        )
        # Update state immutably with video path & frame timestamps
        new_state: AppState = replace(state, video_path=video_path, frame_timestamps_ns=frame_timestamps_ns)
        yield recording.binary_stream().read(), new_state

    # def on_nav(self, state: AppState, kind: Action) -> AppState:
    #     """Handle navigation events."""
    #     new_state: AppState = reduce(state, kind)
    #     return new_state
