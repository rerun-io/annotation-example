from dataclasses import dataclass

import gradio as gr
from gradio_rerun import Rerun

from annotation_example.dbt_ui.controller import Controller
from annotation_example.dbt_ui.state import AppState


@dataclass
class ControlPanel:
    root: gr.Column | None = None
    run_bbox: gr.Button | None = None
    run_kpt: gr.Button | None = None
    status: gr.Markdown | None = None

    def build(self):
        with gr.Column() as self.root:
            gr.Markdown("### Control Panel")
            with gr.Row():
                self.run_bbox = gr.Button("Run BBox Network")
                self.run_kpt = gr.Button("Run Keypoint Detection")

        return self

    def wire(self, ctrl: Controller, state: gr.State | AppState, viewer: Rerun) -> None:
        """Wire the control panel with the controller and state."""
        # when we click
        self.run_bbox.click(fn=ctrl.engine.predict_xyxy, inputs=[state], outputs=[state]).then(
            fn=ctrl.log_state, inputs=[state], outputs=[viewer, state]
        )

        # self.run_kpt.click(
        #     lambda s: ctrl.on_nav(s, Action.NEXT), inputs=[state], outputs=[state], api_name="nav/next"
        # ).then(
        #     ctrl.log_state,
        #     inputs=[state],
        #     outputs=[viewer, state],
        # )
