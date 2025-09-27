from dataclasses import dataclass, field
from itertools import islice
from pathlib import Path
from timeit import default_timer as timer

import numpy as np
import rerun as rr
import rerun.blueprint as rrb
from jaxtyping import Float32, Int, UInt8
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
from simplecv.data.exoego.base_exoego import BaseExoEgoSequence, ExoEgoLabels
from simplecv.data.exoego.hocap import HocapSequence
from simplecv.data.skeleton.coco_133 import (
    COCO_133_ID2NAME,
    COCO_133_LINKS,
)
from simplecv.data.skeleton.mediapipe import MEDIAPIPE_ID2NAME, MEDIAPIPE_LINKS
from simplecv.rerun_log_utils import (
    RerunTyroConfig,
)
from simplecv.video_io import MultiVideoReader
from tqdm import tqdm
from wilor_nano.hand_detection import HandDetector, HandDetectorConfig
from wilor_nano.hand_keypoints import (
    HandKeypointDetectorConfig,
    WilorHandKeypointDetector,
)

from annotation_example.mv_hand_tracker import MultiHandState, MultiViewHandTracker, MultiViewHandTrackerConfig


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
class HandTrackingConfig:
    """CLI configuration controlling dataset selection and calibration behaviour."""

    rr_config: RerunTyroConfig
    """Viewer launch configuration propagated to the Rerun harness."""
    dataset: AnnotatedExoEgoDatasetUnion
    """Dataset specification defining which ego/exo sequence to process."""
    log_labels: bool = False
    """Whether to stream ground-truth labels alongside calibration outputs."""
    max_frames: int | None = None
    """Maximum frames to process for debugging speed; ``None`` processes all."""
    mv_config: MultiViewHandTrackerConfig = field(default_factory=MultiViewHandTrackerConfig)
    """Parameters forwarded to the multi-view tracker."""


def main(config: HandTrackingConfig) -> None:
    start_time: float = timer()
    exoego_sequence: BaseExoEgoSequence = config.dataset.setup()
    if not isinstance(exoego_sequence, HocapSequence):
        msg = "multi-view hand tracking currently supports only `HocapSequence` datasets"
        raise TypeError(msg)

    hocap_sequence: HocapSequence = exoego_sequence
    hocap_labels: ExoEgoLabels | None = hocap_sequence.exoego_labels
    if hocap_labels is None or hocap_labels.mano_stack is None:
        msg = "HocapSequence must expose MANO labels to recover subject betas"
        raise ValueError(msg)

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

    exo_sequence: BaseExoSequence | None = hocap_sequence.exo_sequence
    if exo_sequence is None:
        raise ValueError("Selected dataset does not expose an exocentric camera rig.")

    betas: Float32[ndarray, "10"] = hocap_labels.mano_stack.betas.astype(np.float32, copy=False)
    mv_hand_tracker = MultiViewHandTracker(
        config=config.mv_config,
        hand_detector=hand_detection_engine,
        hand_keypoint_detector=hand_keypoint_engine,
        betas=betas,
        pinhole_param_list=exo_sequence.exo_cam_list,
        parent_log_path=parent_log_path,
    )

    exo_video_readers: MultiVideoReader = exo_sequence.exo_video_readers

    total_frames: int = len(shortest_timestamp)
    if config.max_frames is not None:
        total_frames = min(total_frames, config.max_frames)

    limited_iter = islice(zip(shortest_timestamp, exo_video_readers), total_frames)
    hand_state: MultiHandState = MultiHandState()
    for ts_idx, (ts_nano, rgb_list) in enumerate(tqdm(limited_iter, total=total_frames)):
        rr.set_time(timeline=timeline, duration=ts_nano * 1e-9)
        rgb_batch: UInt8[ndarray, "n_views H W 3"] = np.stack(rgb_list, axis=0)
        hand_state: MultiHandState = mv_hand_tracker(
            rgb_batch=rgb_batch,
            pinhole_param_list=exo_sequence.exo_cam_list,
            hand_state=hand_state,
            recording=config.rr_config.rec_stream,
        )

    print(f"Total time taken: {timer() - start_time:.2f} seconds")
