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
    WilorHandKeypointDetector,
)
from annotation_example.mv_hand_tracker import MultiViewHandTracker, MultiViewHandTrackerConfig


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


@dataclass
class BenchmarkHandCalibConfig:
    """CLI configuration controlling dataset selection and calibration behaviour."""

    rr_config: RerunTyroConfig
    """Viewer launch configuration propagated to the Rerun harness."""
    dataset: AnnotatedExoEgoDatasetUnion
    """Dataset specification defining which ego/exo sequence to process."""
    log_labels: bool = False
    """Whether to stream ground-truth labels alongside calibration outputs."""


def main(config: BenchmarkHandCalibConfig) -> None:
    start_time: float = timer()
    exoego_sequence: BaseExoEgoSequence = config.dataset.setup()  # one-liner
    # only ever accept Hocap dataset for now
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

    mv_hand_tracker = MultiViewHandTracker(
        config=MultiViewHandTrackerConfig(detection_confidence=0.5, verbose=False),
        hand_detector=hand_detection_engine,
        hand_keypoint_detector=hand_keypoint_engine,
        betas=np.zeros((10,), dtype=np.float32),
        parent_log_path=parent_log_path,
    )

    print(f"Total time taken: {timer() - start_time:.2f} seconds")
