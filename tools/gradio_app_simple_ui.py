import gradio as gr

from annotation_example.gradio_ui.super_simple_ui_sink import build

title = """# Detection By Tracking (DBT) for hand pose estimation and annotation"""

with gr.Blocks() as demo:
    gr.Markdown(title)
    build()


if __name__ == "__main__":
    demo.queue(max_size=1).launch(ssr_mode=False)
