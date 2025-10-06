import time
import uuid
from collections.abc import Generator
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import cv2
import gradio as gr
import rerun as rr
import rerun.blueprint as rrb
from gradio_rerun import Rerun
from jaxtyping import UInt8
from numpy import ndarray

RRD_SAVE_DIR: Final[Path] = Path("data") / "rrd-gradio-saves"
RRD_SUFFIX: Final[str] = ".rrd"


def get_recording(recording_id: uuid.UUID | None, application_id: str = "Application ID") -> rr.RecordingStream:
    return rr.RecordingStream(application_id=application_id, recording_id=recording_id)


@dataclass(slots=True)
class SimpleAppState:
    """Session-local recording metadata and buffered stream bytes."""

    recording_id: uuid.UUID
    """Recording identifier reused for incremental logging."""


def _log_img(
    state: SimpleAppState, image: UInt8[ndarray, "height width 3"]
) -> Generator[tuple[SimpleAppState, bytes], None, None]:
    recording: rr.RecordingStream = get_recording(recording_id=state.recording_id)

    stream: rr.BinaryStream = recording.binary_stream()

    RRD_SAVE_DIR.mkdir(parents=True, exist_ok=True)

    blueprint = rrb.Blueprint(
        rrb.Horizontal(
            rrb.Spatial2DView(origin="image/original"),
            rrb.Spatial2DView(origin="image/blurred"),
        ),
        collapse_panels=True,
    )

    recording.send_blueprint(blueprint)
    recording.set_time("iteration", sequence=0)
    recording.log("image/original", rr.Image(image))

    rrd_bytes: bytes | None = stream.read()
    if rrd_bytes is not None:
        yield state, rrd_bytes

    blur: UInt8[ndarray, "height width 3"] = image.copy()
    for i in range(50):
        recording.set_time("iteration", sequence=i)

        # Pretend blurring takes a while so we can see streaming in action.
        time.sleep(0.1)
        blur = cv2.GaussianBlur(blur, (5, 5), 0)
        recording.log("image/blurred", rr.Image(blur))

        # Each time we yield bytes from the stream back to Gradio, they
        # are incrementally sent to the viewer. Make sure to yield any time
        # you want the user to be able to see progress.
        rrd_bytes: bytes | None = stream.read()
        if rrd_bytes is not None:
            yield state, rrd_bytes

    rrd_bytes: bytes | None = stream.read()
    if rrd_bytes is not None:
        yield state, rrd_bytes


# When exposing Gradio callbacks that stream results to the Rerun viewer,
# leave the callback parameters unannotated to avoid Beartype issues.
def log_img(state, image):
    if image is None:
        raise gr.Error("Must provide an image to blur.")

    app_state: SimpleAppState = state
    image_uint8: UInt8[ndarray, "height width 3"] = image

    yield from _log_img(app_state, image_uint8)


def build() -> gr.Blocks:
    session_state: gr.State = gr.State(SimpleAppState(recording_id=uuid.uuid4()))
    with gr.Blocks() as demo:
        with gr.Row():
            with gr.Column(scale=2):
                input_image: gr.Image = gr.Image(
                    label="Input Image",
                    type="numpy",
                    image_mode="RGB",
                )
                gr.Examples(
                    examples=[
                        ["https://huggingface.co/datasets/pablovela5620/minicoco/resolve/main/000000000785.jpg"],
                        ["https://huggingface.co/datasets/pablovela5620/minicoco/resolve/main/000000001296.jpg"],
                    ],
                    inputs=[input_image],
                    cache_examples=False,
                )
                blur_button: gr.Button = gr.Button("Blur Image")
            with gr.Column(scale=5):
                rerun_viewer: Rerun = Rerun(
                    streaming=True,
                    panel_states={
                        "time": "collapsed",
                        "blueprint": "collapsed",
                        "selection": "collapsed",
                    },
                )

        blur_button.click(
            log_img,
            inputs=[session_state, input_image],
            outputs=[session_state, rerun_viewer],
        )

    return demo
