from dataclasses import dataclass

import gradio as gr


@dataclass
class InfoPanel:
    root: gr.Column | None = None

    def build(self):
        with gr.Column() as self.root:
            gr.Markdown("### Info Panel")
            with gr.Row():
                gr.Markdown("Ready.")

        return self
