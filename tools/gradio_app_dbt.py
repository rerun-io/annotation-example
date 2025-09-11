import gradio as gr

from annotation_example.dbt_ui.dbt_main import main

title = """# Detection By Tracking (DBT) for hand pose estimation and annotation"""

with gr.Blocks() as demo:
    gr.Markdown(title)
    main().build()


if __name__ == "__main__":
    demo.queue(max_size=1).launch(ssr_mode=False)
