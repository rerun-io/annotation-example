from dataclasses import dataclass

import gradio as gr
from gradio_rerun import Rerun

from annotation_example.dbt_ui.controller import Controller
from annotation_example.dbt_ui.dbt_callbacks import register_label_keypoint
from annotation_example.dbt_ui.state import AppState


@dataclass
class LabelPanel:
    root: gr.Column | None = None
    hand_selector: gr.Radio | None = None
    bbox_selector: gr.Radio | None = None
    confirm_button: gr.Button | None = None
    status: gr.Markdown | None = None

    def build(self):
        with gr.Column() as self.root:
            gr.Markdown("### Label Panel")
            gr.Markdown(
                "Select a desired timestamp, then click a 2D view to drop a keypoint for that specific frame."
                "  `TL` = top-left corner, `BR` = bottom-right corner. Choose `None` to ignore clicks."
            )
            self.status = gr.Markdown("No keypoints yet.")
            self.hand_selector = gr.Radio(
                label="Hand",
                choices=["Left hand", "Right hand"],
                value="Left hand",
                interactive=True,
            )
            self.bbox_selector = gr.Radio(
                label="Bounding box corner",
                choices=["TL", "BR", "None"],
                value="None",
                interactive=True,
            )
            self.confirm_button = gr.Button("Confirm bounding box", variant="primary")
        return self

    def wire(
        self,
        ctrl: Controller,
        state: gr.State | AppState,
        viewer: Rerun,
        ) -> None:
        self.hand_selector.change(
            ctrl.set_hand_selection,
            inputs=[state, self.hand_selector],
            outputs=[state],
        )
        self.bbox_selector.change(
            ctrl.set_bbox_corner_selection,
            inputs=[state, self.bbox_selector],
            outputs=[state],
        )
        viewer.selection_change(
            register_label_keypoint,
            inputs=[state],
            outputs=[state, self.status],
        ).then(
            ctrl.log_keypoint_clicks,
            inputs=[state],
            outputs=[viewer, state, self.status],
        )
