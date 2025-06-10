import gradio as gr

# 2️⃣  immediately patch the safety helpers
import gradio.processing_utils as _pu

_pu._is_blocked_path = lambda *a, **kw: False  # disables dot-dir veto
_pu._check_allowed = lambda *a, **kw: None  # disables all path checks

from annotation_example.gradio_ui.person_annot import build_annot_ui, img_dir  # noqa: E402

with gr.Blocks() as demo:
    build_annot_ui()


if __name__ == "__main__":
    print(f"Launching demo with allowed paths: {img_dir.as_posix()}")
    allowed_paths = [img_dir.as_posix(), img_dir.parent.as_posix()]
    demo.queue(max_size=2).launch(allowed_paths=allowed_paths)
