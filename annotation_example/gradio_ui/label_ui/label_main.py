import uuid
from pathlib import Path
from typing import cast

import gradio as gr
from gradio_rerun import Rerun

from mv_api.gradio_ui.label_ui.controller import Controller
from mv_api.gradio_ui.label_ui.engine import Engine
from mv_api.gradio_ui.label_ui.label_callbacks import track_current_time
from mv_api.gradio_ui.label_ui.panels.bbox_panel import BoundingBoxPanel
from mv_api.gradio_ui.label_ui.panels.info_panel import InfoPanel
from mv_api.gradio_ui.label_ui.state import AppState

INPUT_TAB_ID: str = "input_tab"
OUTPUT_TAB_ID: str = "output_tab"

EXAMPLE_RRD_FILE_PATHS: list[Path] = [
    # Path("/mnt/8tb/data/exoego-self-collected/gus/statisOrangePNP_av1.rrd"),
    # Path("/mnt/8tb/data/exoego-self-collected/gus/17600630913N_staticRandomCupStack-annotated.rrd"),
]
EXAMPLE_RRD_PATHS_LIST: list[list[str]] = [[str(path)] for path in EXAMPLE_RRD_FILE_PATHS]
EXAMPLE_RRD_ALLOWED_DIRS: list[str] = [str(path.parent) for path in EXAMPLE_RRD_FILE_PATHS]

engine: Engine = Engine() if gr.NO_RELOAD or "engine" not in globals() else cast(Engine, globals()["engine"])


class LabelApp:
    def __init__(
        self,
        *,
        app_name: str = "label_app",
        # engine_cfg: EngineConfig | None = None,
    ):
        self.engine = engine
        self.ctrl = Controller(engine=self.engine)
        self.info_panel = InfoPanel()
        self.bbox_panel = BoundingBoxPanel()
        self._demo: gr.Blocks | None = None

    def build(self):
        self.state_comp = gr.State(AppState(recording_id=uuid.uuid4()))

        with gr.Blocks() as demo, gr.Row():
            with gr.Column(scale=2), gr.Tabs(selected=INPUT_TAB_ID) as io_tabs:
                with gr.TabItem("Input: RRD File", id=INPUT_TAB_ID):
                    with gr.Accordion("Upload RRD File", open=True) as upload_drawer:
                        rrd_file: gr.File = gr.File(file_types=[".rrd"], label="Upload dataset annotated rrd file")
                        # gr.Examples(
                        #     examples=EXAMPLE_RRD_PATHS_LIST,
                        #     inputs=[rrd_file],
                        #     cache_examples=False,
                        # )

                    self.info_panel.build()
                    status_display: gr.Text = cast(gr.Text, self.info_panel.status)
                    self.bbox_panel.build()
                with gr.TabItem("Output: Annotated RRD", id=OUTPUT_TAB_ID):
                    annotated_rrd_file: gr.File = gr.File(label="Download annotated RRD file", interactive=False)
            with gr.Column(scale=5):
                viewer = Rerun(
                    streaming=True,
                    panel_states={
                        "time": "collapsed",
                        "blueprint": "collapsed",
                        "selection": "hidden",
                    },
                    height=850,
                )
                viewer.time_update(track_current_time, inputs=[self.state_comp], outputs=[self.state_comp])

        # Collapse the upload accordion when the video changes
        rrd_file.change(
            fn=lambda _: gr.Accordion(open=False),
            inputs=[rrd_file],
            outputs=[upload_drawer],
        ).then(
            fn=self.info_panel.show_running,
            inputs=[],
            outputs=[status_display],
        ).then(
            fn=self.ctrl.initialize_rrd,
            inputs=[rrd_file, self.state_comp],
            outputs=[viewer, self.state_comp],
        ).then(
            fn=self.info_panel.show_ready,
            inputs=[],
            outputs=[status_display],
        )

        #  wire controls
        self.bbox_panel.wire(
            self.ctrl,
            state=self.state_comp,
            viewer=viewer,
            tabs=io_tabs,
            output_tab_id=OUTPUT_TAB_ID,
            annotated_rrd_file=annotated_rrd_file,
        )

        # self.label_panel.wire(
        #     ctrl=self.ctrl,
        #     state=self.state_comp,
        #     viewer=viewer,
        # )

        return demo


# quick CLI
def main():
    return LabelApp()
