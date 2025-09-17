import uuid

import gradio as gr
from gradio_rerun import Rerun

from annotation_example.dbt_ui.controller import Controller
from annotation_example.dbt_ui.dbt_callbacks import (
    activate_label_panel,
    activate_run_networks_panel,
    track_current_time,
)
from annotation_example.dbt_ui.engine import Engine
from annotation_example.dbt_ui.panels.control_panel import ControlPanel
from annotation_example.dbt_ui.panels.info_panel import InfoPanel
from annotation_example.dbt_ui.panels.label_panel import LabelPanel
from annotation_example.dbt_ui.state import AppState

if gr.NO_RELOAD:
    engine = Engine()


class LabelApp:
    def __init__(
        self,
        *,
        app_name: str = "label_app",
        # engine_cfg: EngineConfig | None = None,
    ):
        self.engine = engine
        self.ctrl = Controller(engine=self.engine)
        self.control_panel = ControlPanel()
        self.info_panel = InfoPanel()
        self.label_panel = LabelPanel()
        self._demo: gr.Blocks | None = None

    def build(self):
        self.state_comp = gr.State(AppState(recording_id=uuid.uuid4()))

        with gr.Blocks() as demo, gr.Row():
            with gr.Column(scale=2):
                with gr.Accordion("Upload Video", open=True) as upload_drawer:
                    zip_file: gr.File = gr.File(file_types=[".zip"], label="Upload dataset .zip")

                    gr.Examples(
                        examples=[
                            ["data/ego-t265-videos.zip"],
                            ["data/egoexo-cardshuffle1-videos.zip"],
                            ["data/egoexo-hocap.zip"],
                            ["data/exo-lg-videos.zip"],
                            # ["data/exo-lg-videos-trimmed.zip"],
                        ],
                        inputs=[zip_file],
                        cache_examples=False,
                    )

                self.info_panel.build()
                with gr.Tabs():
                    with gr.TabItem("Run Networks", id="run-networks") as run_networks_tab:
                        self.control_panel.build()
                    with gr.TabItem("Label", id="label") as label_tab:
                        self.label_panel.build()
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

        # # initial log + status
        demo.load(
            self.ctrl.log_state,
            inputs=[self.state_comp],
            outputs=[viewer, self.state_comp],
        )

        # Collapse the upload accordion when the video changes
        zip_file.change(
            fn=lambda _: gr.Accordion(open=False),
            inputs=[zip_file],
            outputs=[upload_drawer],
        ).then(
            fn=self.ctrl.initialize_rrd,
            inputs=[zip_file, self.state_comp],
            outputs=[viewer, self.state_comp],
        )

        # # wire controls
        self.control_panel.wire(
            self.ctrl,
            state=self.state_comp,
            viewer=viewer,
        )

        self.label_panel.wire(
            ctrl=self.ctrl,
            state=self.state_comp,
            viewer=viewer,
        )

        run_networks_tab.select(
            activate_run_networks_panel,
            inputs=[self.state_comp],
            outputs=[self.state_comp],
        )

        label_tab.select(
            activate_label_panel,
            inputs=[self.state_comp],
            outputs=[self.state_comp],
        )

        # self._demo = demo
        return demo


# quick CLI
def main():
    return LabelApp()
