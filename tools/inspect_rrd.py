#!/usr/bin/env python3
"""Utility for inspecting data logged in Rerun ``.rrd`` files."""

from __future__ import annotations

import argparse
from pathlib import Path

import pyarrow as pa
import rerun as rr


def _summarize_table(table: pa.Table, label: str) -> None:
    row_count: int = table.num_rows
    column_names: list[str] = list(table.column_names)
    print(f"{label}: {row_count} rows")
    if not column_names:
        print("  (no columns)")
        return

    print(f"  columns: {', '.join(column_names)}")
    preview_row_count: int = min(3, row_count)
    if preview_row_count == 0:
        return

    preview: pa.Table = table.slice(0, preview_row_count)
    preview_dict: dict[str, list[object]] = preview.to_pydict()
    for column_name, values in preview_dict.items():
        value_preview: object | None = values[0] if values else None
        formatted_preview: str = _format_value_preview(value_preview)
        print(f"  {column_name}: {formatted_preview}")


def _format_value_preview(value: object | None) -> str:
    if value is None:
        return "None"
    if isinstance(value, (bytes, bytearray)):
        return f"<bytes {len(value)} B>"
    if isinstance(value, list):
        length: int = len(value)
        if value and isinstance(value[0], list):
            inner_length: int = len(value[0])
            suffix: str = ", ..." if length > 1 else ""
            return f"[list(len={inner_length}){suffix}] (len={length})"
        prefix: list[str] = [repr(item) for item in value[:5]]
        suffix: str = ", ..." if length > 5 else ""
        return f"[{', '.join(prefix)}{suffix}] (len={length})"
    if isinstance(value, dict):
        keys: list[str] = list(value.keys())
        preview_keys: list[str] = keys[:5]
        suffix: str = ", ..." if len(keys) > 5 else ""
        return f"{{{', '.join(preview_keys)}{suffix}}}"
    return repr(value)


def _inspect_rrd(rrd_path: Path, index_timeline: str, contents: str) -> None:
    recording: rr.dataframe.Recording = rr.dataframe.load_recording(str(rrd_path))
    view: rr.dataframe.RecordingView = recording.view(index=index_timeline, contents=contents)
    selected: pa.RecordBatchReader = view.select()
    table: pa.Table = selected.read_all()
    print(f"Inspecting {rrd_path}")
    _summarize_table(table, "data")


def _parse_args() -> argparse.Namespace:
    parser: argparse.ArgumentParser = argparse.ArgumentParser(
        description="Inspect the contents of a Rerun .rrd file using the dataframe API.",
    )
    parser.add_argument("paths", nargs="+", type=Path, help="One or more .rrd files to inspect.")
    parser.add_argument(
        "--index",
        default="time",
        help="Timeline name to use as the dataframe index (default: time).",
    )
    parser.add_argument(
        "--contents",
        default="/**",
        help="Entity expression describing which data to include (default: /**).",
    )
    args: argparse.Namespace = parser.parse_args()
    return args


def main() -> None:
    args: argparse.Namespace = _parse_args()
    index_timeline: str = args.index
    contents_expr: str = args.contents
    for path in args.paths:
        resolved_path: Path = path.expanduser().resolve()
        if not resolved_path.exists():
            print(f"Skipping missing file: {resolved_path}")
            continue
        _inspect_rrd(resolved_path, index_timeline, contents_expr)


if __name__ == "__main__":
    main()
