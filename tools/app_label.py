import os
from dataclasses import dataclass, field
from typing import Final

import gradio as gr
import tyro

from annotation_example.gradio_ui.label_ui.label_main import EXAMPLE_RRD_ALLOWED_DIRS, main


def _env_flag(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    lowered = value.strip().lower()
    return lowered in {"1", "true", "yes", "on"}


def _env_allowed_paths(name: str) -> list[str]:
    raw = os.environ.get(name)
    if not raw:
        return []
    return [entry for entry in raw.split(os.pathsep) if entry]


GRADIO_SERVER_NAME_DEFAULT: Final[str] = os.environ.get("GRADIO_SERVER_NAME", "0.0.0.0")
GRADIO_SERVER_PORT_DEFAULT: Final[int] = int(os.environ.get("GRADIO_SERVER_PORT", "7860"))
GRADIO_SHARE_DEFAULT: Final[bool] = _env_flag("GRADIO_SHARE", False)
GRADIO_ROOT_PATH_DEFAULT: Final[str | None] = os.environ.get("GRADIO_ROOT_PATH") or None
LABEL_APP_ALLOWED_PATHS_ENV: Final[list[str]] = _env_allowed_paths("LABEL_APP_ALLOWED_PATHS")


def _default_allowed_paths() -> list[str]:
    merged: list[str] = []
    if LABEL_APP_ALLOWED_PATHS_ENV:
        merged.extend(LABEL_APP_ALLOWED_PATHS_ENV)
    if EXAMPLE_RRD_ALLOWED_DIRS:
        merged.extend(EXAMPLE_RRD_ALLOWED_DIRS)
    # Preserve order while removing duplicates
    return list(dict.fromkeys(merged))


@dataclass
class LabelAppConfig:
    """Runtime configuration for the labeling UI CLI."""

    ssr_mode: bool = False
    """Enable Gradio SSR mode (disabled by default)."""

    use_queue: bool = False
    """Enable the Gradio request queue (disabled to avoid UI lock-ups)."""

    queue_size: int = 1
    """Maximum size of the Gradio request queue when enabled."""

    share: bool = GRADIO_SHARE_DEFAULT
    """Whether to request a public Gradio share link."""

    server_name: str = GRADIO_SERVER_NAME_DEFAULT
    """Hostname/IP address to bind the Gradio server to."""

    server_port: int = GRADIO_SERVER_PORT_DEFAULT
    """Port to expose the Gradio server on."""

    allowed_paths: list[str] = field(default_factory=_default_allowed_paths)
    """Directories that Gradio may access when serving uploaded assets."""

    root_path: str | None = GRADIO_ROOT_PATH_DEFAULT
    """Reverse-proxy mount prefix (set when app is served from a subpath)."""


def run_app(config: LabelAppConfig) -> None:
    """Instantiate and launch the labeling Gradio application."""
    title = """# Labeling app to fix hand keypoints on Exoego RRD files"""

    with gr.Blocks() as demo:
        gr.Markdown(title)
        main().build()

    allowed_paths: list[str] = list(dict.fromkeys(config.allowed_paths))

    if config.use_queue:
        demo = demo.queue(max_size=config.queue_size)

    allowed_paths_arg: list[str] | None = allowed_paths if allowed_paths else None

    demo.launch(
        ssr_mode=config.ssr_mode,
        share=config.share,
        server_name=config.server_name,
        server_port=config.server_port,
        allowed_paths=allowed_paths_arg,
        root_path=config.root_path,
    )


if __name__ == "__main__":
    run_app(tyro.cli(LabelAppConfig))
