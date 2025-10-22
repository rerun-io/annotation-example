from dataclasses import dataclass

import gradio as gr


@dataclass
class InfoPanel:
    """Container for high-level status messaging in the label UI."""

    root: gr.Column | None = None
    """Root column containing info panel components."""

    status: gr.Text | None = None
    """Status text component that reflects pipeline progress."""

    ready_text: str = "Ready."
    """Status text displayed when the pipeline is idle."""

    running_text: str = "🏃 Running..."
    """Status text shown while a new RRD file is processing."""

    def build(self) -> "InfoPanel":
        with gr.Column() as self.root:
            gr.Markdown("### Info Panel")
            with gr.Row():
                self.status = gr.Text(self.ready_text, label="Status", interactive=False)

        return self

    def show_ready(self) -> str:
        """Return the idle status text for Gradio updates."""

        return self.ready_text

    def show_running(self) -> str:
        """Return the in-progress status text for Gradio updates."""

        return self.running_text
