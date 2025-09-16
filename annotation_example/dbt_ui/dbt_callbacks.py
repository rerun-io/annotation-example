from dataclasses import replace

import gradio as gr
import numpy as np
from gradio_rerun.events import (
    SelectionChange,
    TimeUpdate,
)
from rerun.event import SelectionChangeEvent

from annotation_example.dbt_ui.state import AppState


def initial_callback(
    request: gr.Request,
    change: SelectionChange,
):
    evt: SelectionChangeEvent = change.payload


def track_current_time(state: AppState, evt: TimeUpdate) -> AppState:
    timestamp_ns: float = evt.payload.time
    # if greater than 1e6 assume it's in nanoseconds and floor
    time_ns: int = int(np.floor(timestamp_ns))
    # print(f"Time update: {time_ns * 1e-9:.3f} seconds")
    state: AppState = replace(state, current_time_ns=time_ns)
    return state
