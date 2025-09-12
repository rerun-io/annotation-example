import uuid

import gradio as gr
import rerun as rr
from gradio_rerun import Rerun

from annotation_example.dbt_ui.controller import Controller
from annotation_example.dbt_ui.dbt_callbacks import track_current_time
from annotation_example.dbt_ui.engine import Engine
from annotation_example.dbt_ui.panels.control_panel import ControlPanel
from annotation_example.dbt_ui.panels.info_panel import InfoPanel
from annotation_example.dbt_ui.state import AppState


def get_recording(
    recording_id: uuid.UUID, application_id: str = "Detection By Tracking Annotation"
) -> rr.RecordingStream:
    return rr.RecordingStream(application_id=application_id, recording_id=recording_id)


class LabelApp:
    def __init__(
        self,
        *,
        app_name: str = "label_app",
        # engine_cfg: EngineConfig | None = None,
    ):
        self.engine = Engine()
        self.ctrl = Controller(engine=self.engine)
        self.control_panel = ControlPanel()
        self.info_panel = InfoPanel()
        self._demo: gr.Blocks | None = None

    def build(self):
        self.state_comp = gr.State(AppState(recording_id=uuid.uuid4()))

        with gr.Blocks() as demo, gr.Row():
            with gr.Column(scale=2):
                with gr.Accordion("Upload Video", open=True) as upload_drawer:
                    zip_file: gr.File = gr.File(file_types=[".zip"], label="Upload dataset .zip")

                    gr.Examples(
                        examples=[["../../../personal/mv-api/data/multicam-sample/card-shuffle1.zip"]],
                        inputs=[zip_file],
                        cache_examples=False,
                    )

                self.info_panel.build()
                self.control_panel.build()
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
        demo.load(self.ctrl.log_state, inputs=[self.state_comp], outputs=[viewer, self.state_comp])

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
        self.control_panel.wire(self.ctrl, state=self.state_comp, viewer=viewer)

        # self._demo = demo
        return demo


# quick CLI
def main():
    return LabelApp()
