from dataclasses import dataclass, field
from pathlib import Path
from timeit import default_timer as timer
from typing import NamedTuple

import cv2
import numpy as np
import rerun as rr
import rerun.blueprint as rrb
from jaxtyping import Float, Int, UInt8
from numpy import ndarray
from simplecv.apis.view_exoego import (
    LogPaths,
    SceneSetupResult,
    create_blueprint,
    log_exoego_batch,
    setup_scene,
)
from simplecv.configs.exoego_dataset_configs import AnnotatedExoEgoDatasetUnion
from simplecv.data.exo.base_exo import BaseExoSequence
from simplecv.data.exoego.base_exoego import BaseExoEgoSequence
from simplecv.data.skeleton.coco_133 import (
    COCO_133_ID2NAME,
    COCO_133_LINKS,
)
from simplecv.data.skeleton.mediapipe import MEDIAPIPE_ID2NAME, MEDIAPIPE_LINKS
from simplecv.rerun_log_utils import (
    RerunTyroConfig,
)
from simplecv.video_io import MultiVideoReader
from wilor_nano.hand_detection import HandDetector, HandDetectorConfig
from wilor_nano.hand_keypoints import (
    HandKeypointDetectorConfig,
    KeypointResults,
    WilorHandKeypointDetector,
)

from annotation_example.hand_calibrator import HandCalibrator, HandCalibratorConfig


def timestamp_to_frame_index(time_ns: int, frame_timestamps_ns: Int[ndarray, "num_frames"]) -> int:
    """Return the frame index at or before ``time_ns`` for monotonic timestamps."""

    idx: int = int(np.searchsorted(frame_timestamps_ns, time_ns, side="right") - 1)
    return max(0, min(idx, len(frame_timestamps_ns) - 1))


def frame_index_to_timestamp(frame_timestamps_ns: Int[ndarray, "num_frames"], frame_index: int) -> int:
    """Return the nanosecond timestamp associated with ``frame_index``."""

    if frame_index < 0 or frame_index >= int(frame_timestamps_ns.shape[0]):
        msg = f"frame_index {frame_index} is outside the valid range [0, {frame_timestamps_ns.shape[0] - 1}]"
        raise IndexError(msg)
    timestamp_ns: int = int(frame_timestamps_ns[frame_index])
    return timestamp_ns


def set_annotation_context(recording: rr.RecordingStream | None = None) -> None:
    """Register Mediapipe (per-hand) and COCO-133 annotation metadata with Rerun."""

    rr.log(
        "/",
        rr.AnnotationContext(
            [
                rr.ClassDescription(
                    info=rr.AnnotationInfo(id=0, label="L", color=(0, 0, 255)),
                    keypoint_annotations=[
                        rr.AnnotationInfo(id=kp_id, label=name) for kp_id, name in MEDIAPIPE_ID2NAME.items()
                    ],
                    keypoint_connections=MEDIAPIPE_LINKS,
                ),
                rr.ClassDescription(
                    info=rr.AnnotationInfo(id=1, label="R", color=(255, 0, 0)),
                    keypoint_annotations=[
                        rr.AnnotationInfo(id=kp_id, label=name) for kp_id, name in MEDIAPIPE_ID2NAME.items()
                    ],
                    keypoint_connections=MEDIAPIPE_LINKS,
                ),
                rr.ClassDescription(
                    info=rr.AnnotationInfo(id=2, label="Coco Wholebody", color=(0, 0, 255)),
                    keypoint_annotations=[
                        rr.AnnotationInfo(id=kp_id, label=name) for kp_id, name in COCO_133_ID2NAME.items()
                    ],
                    keypoint_connections=COCO_133_LINKS,
                ),
            ]
        ),
        static=True,
        recording=recording,
    )


class ParsedDetections(NamedTuple):
    """Convenience container bundling multi-view detections for triangulation."""

    uvc_coco_batch: Float[ndarray, "n_views n_kpts=133 3"]
    right_hand_kpts: KeypointResults | None
    left_hand_kpts: KeypointResults | None


@dataclass
class BenchmarkHandCalibConfig:
    """CLI configuration controlling dataset selection and calibration behaviour."""

    rr_config: RerunTyroConfig
    """Viewer launch configuration propagated to the Rerun harness."""
    dataset: AnnotatedExoEgoDatasetUnion
    """Dataset specification defining which ego/exo sequence to process."""
    log_labels: bool = False
    """Whether to stream ground-truth labels alongside calibration outputs."""
    hand_calibrator: HandCalibratorConfig = field(default_factory=HandCalibratorConfig)
    """Parameter bundle forwarded to the ``HandCalibrator`` instance."""


def mv_reader_to_rgb_ts_batch(
    mv_reader: MultiVideoReader,
    num_frames: int,
    ts_nanos: int,
    frame_timestamps_ns: Int[ndarray, "n_frames"],
) -> UInt8[ndarray, "n_frames n_views H W 3"]:
    """Slice a timestamp-aligned RGB batch from a ``MultiVideoReader``.

    Args:
        mv_reader: Multi-camera video reader covering the synchronized views.
        num_frames: Number of consecutive frames to fetch, inclusive of the
            frame containing ``ts_nanos``.
        ts_nanos: Absolute nanosecond timestamp selecting the first frame.
        frame_timestamps_ns: Shared monotonic timestamp vector used to align
            frame indices across views (commonly ``SceneSetupResult.shortest_timestamp``).

    Returns:
        ``UInt8[np.ndarray, "n_frames n_views H W 3"]`` containing the RGB
        frames ordered by increasing timestamp and view index.

    Raises:
        ValueError: If there are no views, no frames, ``num_frames`` is not
            positive, or the requested batch exceeds the available frames from
            ``ts_nanos`` onward.
    """

    if num_frames <= 0:
        raise ValueError("num_frames must be a positive integer")

    total_frames: int = len(mv_reader)
    n_views: int = len(mv_reader.video_readers)
    if total_frames == 0 or n_views == 0:
        raise ValueError("MultiVideoReader contains no frames or no views")

    start_idx: int = timestamp_to_frame_index(ts_nanos, frame_timestamps_ns)
    max_available_frames: int = total_frames - start_idx
    if max_available_frames < num_frames:
        raise ValueError("Requested number of frames exceeds available frames from the provided timestamp")

    rgb_frames: list[UInt8[ndarray, "n_views H W 3"]] = []
    for frame_offset in range(num_frames):
        frame_idx: int = start_idx + frame_offset
        bgr_list: list[UInt8[ndarray, "H W 3"]] = mv_reader[frame_idx]
        rgb_views: list[UInt8[ndarray, "H W 3"]] = [cv2.cvtColor(bgr_hw3, cv2.COLOR_BGR2RGB) for bgr_hw3 in bgr_list]
        rgb_views_stack: UInt8[ndarray, "n_views H W 3"] = np.stack(rgb_views, axis=0)
        rgb_frames.append(rgb_views_stack)

    rgb_ts_batch: UInt8[ndarray, "n_frames n_views H W 3"] = np.stack(rgb_frames, axis=0)
    return rgb_ts_batch


def main(config: BenchmarkHandCalibConfig) -> None:
    start_time: float = timer()
    exoego_sequence: BaseExoEgoSequence = config.dataset.setup()  # one-liner
    rr.log("/", exoego_sequence.world_coordinate_system, static=True)
    set_annotation_context()

    parent_log_path = Path("world")
    timeline: str = "video_time"

    scene_setup_result: SceneSetupResult = setup_scene(exoego_sequence, parent_log_path, timeline)
    log_paths: LogPaths = scene_setup_result.log_paths
    shortest_timestamp: Int[ndarray, "n_frames"] = scene_setup_result.shortest_timestamp

    blueprint: rrb.Blueprint = create_blueprint(
        exo_video_log_paths=log_paths.exo_video_log_paths,
        ego_video_log_paths=log_paths.ego_video_log_paths,
    )
    rr.send_blueprint(blueprint)

    if config.log_labels:
        log_exoego_batch(
            exoego_sequence=exoego_sequence,
            timeline=timeline,
            shortest_timestamp=shortest_timestamp,
            parent_log_path=parent_log_path,
        )

    hand_detection_engine = HandDetector(HandDetectorConfig(verbose=False))
    hand_keypoint_engine = WilorHandKeypointDetector(HandKeypointDetectorConfig(verbose=False))

    exo_sequence: BaseExoSequence | None = exoego_sequence.exo_sequence
    if exo_sequence is None:
        raise ValueError("Selected dataset does not expose an exocentric camera rig.")

    target_ts_nano: int = (
        config.hand_calibrator.ts_nano
        if config.hand_calibrator.ts_nano is not None
        else frame_index_to_timestamp(shortest_timestamp, 0)
    )
    frame_index: int = timestamp_to_frame_index(target_ts_nano, shortest_timestamp)
    frame_timestamp_ns: int = frame_index_to_timestamp(shortest_timestamp, frame_index)
    frame_timestamp_seconds: float = frame_timestamp_ns * 1e-9
    rr.set_time(timeline=timeline, duration=frame_timestamp_seconds)

    rgb_ts_batch: UInt8[ndarray, "n_frames n_views H W 3"] = mv_reader_to_rgb_ts_batch(
        mv_reader=exo_sequence.exo_video_readers,
        num_frames=config.hand_calibrator.n_frame_optim,
        ts_nanos=target_ts_nano,
        frame_timestamps_ns=shortest_timestamp,
    )
    hand_calibrator = HandCalibrator(
        hand_detector=hand_detection_engine,
        hand_keypoint_detector=hand_keypoint_engine,
        config=config.hand_calibrator,
        parent_log_path=parent_log_path,
    )
    hand_calibrator(
        exo_cam_list=exo_sequence.exo_cam_list,
        rgb_ts_batch=rgb_ts_batch,
        recording=config.rr_config.rec_stream,
    )

    print(f"Total time taken: {timer() - start_time:.2f} seconds")
