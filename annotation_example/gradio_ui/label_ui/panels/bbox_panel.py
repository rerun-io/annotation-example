from dataclasses import dataclass

import gradio as gr
from gradio_rerun import Rerun

from mv_api.gradio_ui.label_ui.controller import Controller
from mv_api.gradio_ui.label_ui.label_callbacks import register_label_keypoint


@dataclass
class BoundingBoxPanel:
    """Simple controls for selecting hand and corner while annotating bounding boxes."""

    root: gr.Column | None = None
    hand_selector: gr.Radio | None = None
    corner_selector: gr.Radio | None = None
    confirm_button: gr.Button | None = None
    clear_keypoints_button: gr.Button | None = None
    save_rrd_button: gr.Button | None = None

    def build(self) -> "BoundingBoxPanel":
        with gr.Column() as self.root:
            gr.Markdown("### Bounding Box Controls")
            self.hand_selector = gr.Radio(
                choices=["Left Hand", "Right Hand"],
                value="Left Hand",
                label="Select hand",
            )
            self.corner_selector = gr.Radio(
                choices=["Top Left", "Bottom Right"],
                value="Top Left",
                label="Select corner",
            )
            self.confirm_button = gr.Button(value="Confirm Bounding Box", variant="primary")
            self.clear_keypoints_button = gr.Button(value="Clear Current Keypoints", variant="secondary")
            self.save_rrd_button = gr.Button(value="Save Annotated RRD", variant="secondary")
        return self

    def wire(
        self,
        ctrl: Controller,
        state: gr.State,
        viewer: Rerun,
        tabs: gr.Tabs,
        output_tab_id: str,
        annotated_rrd_file: gr.File,
    ) -> None:
        # self.hand_selector.change(
        #     ctrl.set_hand_selection,
        #     inputs=[state, self.hand_selector],
        #     outputs=[state],
        # )
        # self.bbox_selector.change(
        #     ctrl.set_bbox_corner_selection,
        #     inputs=[state, self.bbox_selector],
        #     outputs=[state],
        # ).then(
        #     ctrl.on_corner_selection_changed,
        #     inputs=[state],
        #     outputs=[viewer, state, self.status, self.bbox_selector],
        # )

        viewer.selection_change(
            register_label_keypoint,
            inputs=[state],
            outputs=[state],
        ).then(
            ctrl.log_bbox_kpts,
            inputs=[state, self.corner_selector, self.hand_selector],
            outputs=[viewer, state, self.corner_selector],
        )
        self.confirm_button.click(
            ctrl.confirm_bbox,
            inputs=[state, self.hand_selector],
            outputs=[viewer, state, self.corner_selector],
        )
        self.clear_keypoints_button.click(
            ctrl.clear_current_keypoints,
            inputs=[state],
            outputs=[viewer, state],
        )

        def _select_output_tab() -> gr.Tabs:
            return gr.Tabs(selected=output_tab_id)

        self.save_rrd_button.click(_select_output_tab, inputs=None, outputs=[tabs]).then(
            ctrl.save_annotated_rrd,
            inputs=[state],
            outputs=[state, annotated_rrd_file],
        )
