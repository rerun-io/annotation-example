from dataclasses import dataclass

import gradio as gr
from gradio_rerun import Rerun

from annotation_example.dbt_ui.dbt_callbacks import register_label_keypoint
from annotation_example.dbt_ui.state import AppState


@dataclass
class LabelPanel:
    root: gr.Column | None = None
    clear_button: gr.Button | None = None
    status: gr.Markdown | None = None

    def build(self):
        with gr.Column() as self.root:
            gr.Markdown("### Label Panel")
            gr.Markdown("Select a desired timestamp, then click a 2D view to drop a keypoint for that specific frame.")
            self.status = gr.Markdown("No keypoints yet.")
            self.clear_button = gr.Button("Clear keypoints")
        return self

    def wire(
        self,
        state: gr.State | AppState,
        viewer: Rerun,
    ) -> None:
        viewer.selection_change(
            register_label_keypoint,
            inputs=[state],
            outputs=[viewer, state, self.status],
        )
