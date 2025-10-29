from collections.abc import Generator
from dataclasses import replace
from uuid import UUID

import gradio as gr
import numpy as np
from gradio_rerun.events import (
    SelectionChange,
    TimeUpdate,
)
from rerun.event import (
    EntitySelectionItem,
    SelectionChangeEvent,
    SelectionItem,
)

from annotation_example.gradio_ui.label_ui.state import AppState


def track_current_time(state: AppState, evt: TimeUpdate) -> AppState:
    timestamp_ns: float = evt.payload.time
    # if greater than 1e6 assume it's in nanoseconds and floor
    time_ns: int = int(np.floor(timestamp_ns))
    state: AppState = replace(state, current_time_ns=time_ns)
    return state


def register_label_keypoint(
    state: AppState,
    _request: gr.Request,
    change: SelectionChange,
) -> Generator[AppState, None, None]:
    """Capture the latest viewer selection into state when a valid RGB entity is clicked.

    Args:
        state: Current immutable application state shared between callbacks.
        _request: Gradio request metadata (unused; required by the callback signature).
        change: Selection change event emitted by the Rerun viewer.

    Yields:
        The original state with `selection_evt` cleared for ignored snapshots, or a copy
        carrying the accepted `SelectionChangeEvent` when it passes all filters.
    """
    if not isinstance(state, AppState):
        yield state
        return

    evt: SelectionChangeEvent = change.payload
    if len(evt.items) != 1:
        # Viewer often emits intermediate clears/updates; clear the stored event for those.
        cleared_state: AppState = replace(state, selection_evt=None)
        yield cleared_state
        return

    items: list[SelectionItem] = evt.items
    if len(items) != 1:
        raise gr.Error("Expected exactly one selection item.")
    item: SelectionItem = items[0]
    if not isinstance(item, EntitySelectionItem):
        # Non-entity selections (e.g., view-level) should not propagate downstream.
        cleared_state = replace(state, selection_evt=None)
        yield cleared_state
        return
    # only accept if rgb in entity_path
    if "rgb" not in item.entity_path:
        # Ignore selections outside RGB feeds to avoid mixing modalities.
        cleared_state = replace(state, selection_evt=None)
        yield cleared_state
        return
    # Successful path: forward the event so later callbacks can log it.
    updated_state: AppState = replace(state, selection_evt=evt)

    yield updated_state
