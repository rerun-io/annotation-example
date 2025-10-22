<!--
First look if there is already a similar feature request. If there is, upvote the issue with 👍
-->

**Is your feature request related to a problem? Please describe.**
While wiring the `gradio_rerun.Rerun` component into `annotation_example/gradio_ui/super_simple_ui_sink.py`, I need to stream a recording to the browser and also let the user save the same session as an `.rrd` without re-running the pipeline. The `RecordingStream.binary_stream()` bytes are consumed as soon as they are yielded to Gradio, so there is no supported way to tee the data to both the viewer and disk. I either lose the ability to stream, or I have to duplicate all logging work manually.

**Describe the solution you'd like**
Please extend the upcoming multi-sink API (e.g. https://github.com/rerun-io/rerun/pull/10158) so that a `RecordingStream` can fan out the same data to a Gradio binary stream and to another sink that I can persist (RRD file, callback, etc.) without duplicating every log call. Ideally this would be opt-in when the recording is created, and downstream code could stay unchanged.

**Describe alternatives you've considered**
- Keep the current single-stream setup (works for live viewing but gives me no RRD to save unless I rerun everything).
- Buffer every chunk inside the Gradio state following the suggestion to copy `stream.read()` results before yielding (blows up memory and stalls the queue once images get large).
- Create two independent `RecordingStream` objects, log every entity twice, and manually stitch the second stream into an `.rrd` (works, but doubles CPU usage and introduces easy-to-miss divergence between the streams).

**Additional context**
Attempt 1 – viewer only (shipping code today):

```python
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
        time.sleep(0.1)
        blur = cv2.GaussianBlur(blur, (5, 5), 0)
        recording.log("image/blurred", rr.Image(blur))

        rrd_bytes = stream.read()
        if rrd_bytes is not None:
            yield state, rrd_bytes

    rrd_bytes = stream.read()
    if rrd_bytes is not None:
        yield state, rrd_bytes
```

Attempt 2 – buffer bytes in session state (rejected because the state payload balloons to hundreds of MB and stalls the queue with real recordings). The `gr.State` object is initialised as `gr.State(SimpleAppState(recording_id=uuid.uuid4(), rrd_buffer=bytearray()))`.

```python
@dataclass(slots=True)
class SimpleAppState:
    """Session-local recording metadata and buffered stream bytes."""

    recording_id: uuid.UUID
    """Recording identifier reused for incremental logging."""
    rrd_buffer: bytearray
    """In-memory copy of every chunk streamed to the viewer."""

def _log_img_state_buffer(
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

    chunk: bytes | None = stream.read()
    if chunk is not None:
        state.rrd_buffer.extend(chunk)
        yield state, chunk

    blur: UInt8[ndarray, "height width 3"] = image.copy()
    for i in range(50):
        recording.set_time("iteration", sequence=i)
        time.sleep(0.1)
        blur = cv2.GaussianBlur(blur, (5, 5), 0)
        recording.log("image/blurred", rr.Image(blur))

        chunk = stream.read()
        if chunk is not None:
            state.rrd_buffer.extend(chunk)
            yield state, chunk

    if state.rrd_buffer:
        rrd_path: Path = RRD_SAVE_DIR / f"{state.recording_id}{RRD_SUFFIX}"
        rrd_path.write_bytes(bytes(state.rrd_buffer))
```

Attempt 3 – dual streams (current workaround, but it doubles every log call and risks divergence). The `gr.State` object is still seeded with `SimpleAppState(recording_id=uuid.uuid4())`.

```python
def _log_img_dual_streams(
    state: SimpleAppState, image: UInt8[ndarray, "height width 3"]
) -> Generator[tuple[SimpleAppState, bytes], None, None]:
    viewer_recording: rr.RecordingStream = get_recording(recording_id=state.recording_id)
    viewer_stream: rr.BinaryStream = viewer_recording.binary_stream()

    save_recording: rr.RecordingStream = get_recording(
        recording_id=None, application_id="Gradio RRD Saver"
    )
    save_stream: rr.BinaryStream = save_recording.binary_stream()
    buffered_rrd: bytearray = bytearray()
    RRD_SAVE_DIR.mkdir(parents=True, exist_ok=True)

    blueprint = rrb.Blueprint(
        rrb.Horizontal(
            rrb.Spatial2DView(origin="image/original"),
            rrb.Spatial2DView(origin="image/blurred"),
        ),
        collapse_panels=True,
    )
    viewer_recording.send_blueprint(blueprint)
    save_recording.send_blueprint(blueprint)

    viewer_recording.set_time("iteration", sequence=0)
    save_recording.set_time("iteration", sequence=0)
    viewer_recording.log("image/original", rr.Image(image))
    save_recording.log("image/original", rr.Image(image))

    viewer_chunk: bytes | None = viewer_stream.read()
    if viewer_chunk is not None:
        yield state, viewer_chunk
    save_chunk: bytes | None = save_stream.read()
    if save_chunk is not None:
        buffered_rrd.extend(save_chunk)

    blur: UInt8[ndarray, "height width 3"] = image.copy()
    for i in range(50):
        viewer_recording.set_time("iteration", sequence=i)
        save_recording.set_time("iteration", sequence=i)
        time.sleep(0.1)
        blur = cv2.GaussianBlur(blur, (5, 5), 0)
        viewer_recording.log("image/blurred", rr.Image(blur))
        save_recording.log("image/blurred", rr.Image(blur))

        viewer_chunk = viewer_stream.read()
        if viewer_chunk is not None:
            yield state, viewer_chunk
        save_chunk = save_stream.read()
        if save_chunk is not None:
            buffered_rrd.extend(save_chunk)

    if buffered_rrd:
        rrd_path: Path = RRD_SAVE_DIR / f"{state.recording_id}{RRD_SUFFIX}"
        rrd_path.write_bytes(bytes(buffered_rrd))
```

Without multi-sink support for binary streams I need to keep the dual-recording workaround, and the project incurs the performance hit of logging every entity twice just to produce an `.rrd` alongside the Gradio viewer.
