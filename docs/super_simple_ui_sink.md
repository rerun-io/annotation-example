# Super Simple UI Sink – Single Recording Notes

## Updated implementation

We now keep a single `RecordingStream` per `recording_id` and stream to both the
embedded Gradio viewer and disk:

- Call `set_sinks(grpc_sink)` with a single, reused `rr.GrpcSink()` instance to bootstrap the viewer.
- Immediately open the `binary_stream()` (this wires the viewer drain to the
  gRPC sink).
- Add the file sink with a second `set_sinks(grpc_sink, rr.FileSink(...))`.

All blur iterations log to the same recording, flush once, and drain the viewer
stream. This keeps the viewer responsive and produces `.rrd` files with the full
sequence.

## What changed?

- Setting both sinks **before** calling `binary_stream()` still results in empty
  `.rrd` files (`tools/inspect_rrd.py` reports 0 rows).
- Creating the gRPC sink first, opening the binary stream, and then adding the
  file sink yields `.rrd` files with all frames (`iteration=0..30`).
- Separate recordings are no longer required.

Observed behaviour suggests that creating the binary stream while the file sink
is attached prevents that sink from seeing later data. Re-applying
`set_sinks(...)` after `binary_stream()` while reusing the same `GrpcSink`
instance brings the file sink into the pipeline without interrupting the viewer.

## Verification commands

```
pixi run -e dev python - <<'PY'
import numpy as np, uuid
from annotation_example.gradio_ui.super_simple_ui_sink import _log_img

image = (np.random.rand(16,16,3)*255).astype('uint8')
recording_id = uuid.uuid4()
list(_log_img(recording_id, image))
print(recording_id)
PY

# Replace <recording_id> with the printed UUID
pixi run -e dev python tools/inspect_rrd.py --index iteration \
  data/rrd-gradio-saves/<recording_id>.rrd
```

Expect `data: 31 rows` with `/image/original` and `/image/blurred` columns.
