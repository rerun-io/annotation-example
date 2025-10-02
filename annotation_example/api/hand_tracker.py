from dataclasses import dataclass, field
from itertools import islice
from pathlib import Path
from timeit import default_timer as timer

import numpy as np
import rerun as rr
import rerun.blueprint as rrb
from jaxtyping import Bool, Float32, Int, UInt8
from numpy import ndarray
from simplecv.apis.view_exoego import (
    LogPaths,
    SceneSetupResult,
    compute_vertex_normals_batch,
    create_blueprint,
    log_exoego_batch,
    setup_scene,
)
from simplecv.camera_parameters import Intrinsics, PinholeParameters
from simplecv.configs.exoego_dataset_configs import AnnotatedExoEgoDatasetUnion
from simplecv.data.exo.base_exo import BaseExoSequence
from simplecv.data.exoego.base_exoego import BaseExoEgoSequence, ExoEgoLabels
from simplecv.data.exoego.hocap import HocapSequence
from simplecv.data.skeleton.coco_133 import (
    COCO_133_ID2NAME,
    COCO_133_LINKS,
)
from simplecv.data.skeleton.mediapipe import MEDIAPIPE_ID2NAME, MEDIAPIPE_IDS, MEDIAPIPE_LINKS
from simplecv.rerun_log_utils import (
    Points2DWithConfidence,
    Points3DWithConfidence,
    RerunTyroConfig,
    confidence_scores_to_rgb,
)
from simplecv.video_io import MultiVideoReader
from tqdm import tqdm
from wilor_nano.hand_detection import HandDetector, HandDetectorConfig
from wilor_nano.hand_keypoints import (
    HandKeypointDetectorConfig,
    WilorHandKeypointDetector,
)

from annotation_example.mv_hand_tracker import (
    HAND_LABELS,
    ManoHistory,
    ManoResults,
    MultiHandState,
    MultiViewHandTracker,
    MultiViewHandTrackerConfig,
)


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


def log_mano_outputs(
    *,
    hand_state: MultiHandState,
    tracker: MultiViewHandTracker,
    parent_log_path: Path,
    pinhole_param_list: list[PinholeParameters],
    recording: rr.RecordingStream | None = None,
) -> None:
    """Log optimized MANO meshes together with 3D keypoints and per-view projections."""
    # log the triangulated coco 133 keypoints if available
    if np.any(np.isfinite(hand_state.xyz_coco)):
        conf_colors_3d: UInt8[ndarray, "1 133 3"] = confidence_scores_to_rgb(np.ones((1, 133, 1), dtype=np.float32))
        rr.log(
            str(parent_log_path / "mano_fits/coco_133/keypoints_3d"),
            Points3DWithConfidence(
                positions=hand_state.xyz_coco[0],
                confidences=np.ones((133,), dtype=np.float32),
                class_ids=2,
                keypoint_ids=list(COCO_133_ID2NAME.keys()),
                show_labels=False,
                colors=conf_colors_3d[0],
            ),
            recording=recording,
        )

    for hand_label in HAND_LABELS:
        mano_history: ManoHistory = getattr(hand_state, hand_label)
        mano_result: ManoResults | None = mano_history.t_mano
        if mano_result is None:
            continue

        mano_layer = tracker.left_mano_layer if hand_label == "left" else tracker.right_mano_layer
        class_id: int = 0 if hand_label == "left" else 1
        global_orient: Float32[ndarray, "3"] = mano_result.global_orient.astype(np.float32, copy=False)
        hand_pose: Float32[ndarray, "45"] = mano_result.hand_pose.astype(np.float32, copy=False)
        so3_components: Float32[ndarray, "48"] = np.concatenate(
            [global_orient, hand_pose],
            axis=0,
        )
        so3_batch: Float32[ndarray, "1 48"] = so3_components[np.newaxis, :]
        betas_single: Float32[ndarray, "10"] = mano_result.betas.astype(np.float32, copy=False)
        betas_batch: Float32[ndarray, "1 10"] = betas_single[np.newaxis, :]
        translation_single: Float32[ndarray, "3"] = mano_result.translation.astype(np.float32, copy=False)
        trans_batch: Float32[ndarray, "1 3"] = translation_single[np.newaxis, :]

        mano_mesh: tuple[
            Float32[ndarray, "1 n_verts=778 3"],
            Float32[ndarray, "1 joints_and_tips=21 3"],
        ] = mano_layer(
            th_pose_coeffs=so3_batch,
            th_betas=betas_batch,
            th_trans=trans_batch,
        )
        verts_mm: Float32[ndarray, "n_verts=778 3"] = mano_mesh[0][0].astype(np.float32, copy=False)
        verts_m: Float32[ndarray, "n_verts=778 3"] = (verts_mm / 1000.0).astype(np.float32, copy=False)
        joints_mm: Float32[ndarray, "mp_kpts=21 3"] = mano_mesh[1][0].astype(np.float32, copy=False)
        joints_m: Float32[ndarray, "mp_kpts=21 3"] = (joints_mm / np.float32(1000.0)).astype(
            np.float32,
            copy=False,
        )
        faces_np: Int[ndarray, "n_faces=1538 3"] = mano_layer.th_faces.astype(np.int32, copy=False)
        normals_batch: Float32[ndarray, "1 n_verts=778 3"] = compute_vertex_normals_batch(
            verts_m[np.newaxis, ...],
            faces_np,
        ).astype(np.float32, copy=False)
        normals: Float32[ndarray, "n_verts=778 3"] = normals_batch[0]
        mano_mesh_path: Path = parent_log_path / "mano_fits" / hand_label
        rr.log(
            str(mano_mesh_path),
            rr.Mesh3D(
                vertex_positions=verts_m,
                triangle_indices=faces_np,
                vertex_normals=normals,
                albedo_factor=(64, 128, 255, 255) if hand_label == "left" else (255, 128, 64, 255),
            ),
            recording=recording,
        )

        confidences_3d: Float32[ndarray, "mp_kpts=21"] = np.ones((joints_m.shape[0],), dtype=np.float32)
        conf_colors_3d: UInt8[ndarray, "1 mp_kpts=21 3"] = confidence_scores_to_rgb(
            confidences_3d[np.newaxis, :, np.newaxis]
        )
        rr.log(
            f"{mano_mesh_path}/keypoints_3d",
            Points3DWithConfidence(
                positions=joints_m,
                confidences=confidences_3d,
                class_ids=class_id,
                keypoint_ids=MEDIAPIPE_IDS,
                show_labels=False,
                colors=conf_colors_3d[0],
            ),
            recording=recording,
        )

        ones_column: Float32[ndarray, "mp_kpts=21 1"] = np.ones((joints_m.shape[0], 1), dtype=np.float32)
        world_keypoints_h: Float32[ndarray, "mp_kpts=21 4"] = np.concatenate(
            [joints_m, ones_column],
            axis=1,
        )

        for view_idx, pinhole_param in enumerate(pinhole_param_list):
            camera_name: str = getattr(pinhole_param, "name", f"camera_{view_idx}")
            pinhole_log_path: Path = parent_log_path / "exo" / camera_name / "pinhole"
            hand_video_path: Path = pinhole_log_path / "video" / hand_label
            projection_matrix: Float32[ndarray, "3 4"] = pinhole_param.projection_matrix.astype(
                np.float32,
                copy=False,
            )
            pixel_h: Float32[ndarray, "mp_kpts=21 3"] = (projection_matrix @ world_keypoints_h.T).T.astype(
                np.float32,
                copy=False,
            )
            depths: Float32[ndarray, "mp_kpts=21"] = pixel_h[:, 2]
            positive_depth_mask: Bool[ndarray, "mp_kpts=21"] = depths > 0.0
            uv: Float32[ndarray, "mp_kpts=21 2"] = np.full((world_keypoints_h.shape[0], 2), np.nan, dtype=np.float32)
            np.divide(
                pixel_h[:, :2],
                depths[:, np.newaxis],
                out=uv,
                where=positive_depth_mask[:, np.newaxis],
            )
            finite_uv_mask: Bool[ndarray, "mp_kpts=21"] = (
                np.isfinite(uv[:, 0]) & np.isfinite(uv[:, 1]) & positive_depth_mask
            )
            confidences_2d: Float32[ndarray, "mp_kpts=21"] = positive_depth_mask.astype(np.float32)
            confidences_2d[~finite_uv_mask] = 0.0
            conf_colors_2d: UInt8[ndarray, "1 mp_kpts=21 3"] = confidence_scores_to_rgb(
                confidences_2d[np.newaxis, :, np.newaxis]
            )

            rr.log(
                f"{hand_video_path}/mano_keypoints",
                Points2DWithConfidence(
                    positions=uv,
                    confidences=confidences_2d,
                    class_ids=class_id,
                    keypoint_ids=MEDIAPIPE_IDS,
                    show_labels=False,
                    colors=conf_colors_2d[0],
                ),
                recording=recording,
            )

            xyxy_view: Float32[ndarray, "1 4"]
            if not np.any(finite_uv_mask):
                xyxy_view = np.full((1, 4), np.nan, dtype=np.float32)
            else:
                valid_uv: Float32[ndarray, "m 2"] = uv[finite_uv_mask]
                min_xy: Float32[ndarray, "2"] = valid_uv.min(axis=0).astype(np.float32, copy=False)
                max_xy: Float32[ndarray, "2"] = valid_uv.max(axis=0).astype(np.float32, copy=False)
                width: float = float(max_xy[0] - min_xy[0])
                height: float = float(max_xy[1] - min_xy[1])
                side_length: float = max(width, height, 1.0)
                side_length *= 1.5
                center_xy: Float32[ndarray, "2"] = ((min_xy + max_xy) / np.float32(2.0)).astype(
                    np.float32,
                    copy=False,
                )
                half_side: float = side_length / 2.0
                x1: float = float(center_xy[0] - half_side)
                y1: float = float(center_xy[1] - half_side)
                x2: float = float(center_xy[0] + half_side)
                y2: float = float(center_xy[1] + half_side)

                intrinsics: Intrinsics = pinhole_param.intrinsics
                if intrinsics.width is not None:
                    width_limit: float = float(intrinsics.width)
                    x1 = float(np.clip(x1, 0.0, width_limit))
                    x2 = float(np.clip(x2, 0.0, width_limit))
                if intrinsics.height is not None:
                    height_limit: float = float(intrinsics.height)
                    y1 = float(np.clip(y1, 0.0, height_limit))
                    y2 = float(np.clip(y2, 0.0, height_limit))

                if x1 > x2:
                    x1, x2 = x2, x1
                if y1 > y2:
                    y1, y2 = y2, y1

                xyxy_view = np.array([[x1, y1, x2, y2]], dtype=np.float32)

            rr.log(
                f"{pinhole_log_path}/{hand_label}_mano_bbox",
                rr.Boxes2D(
                    array=xyxy_view,
                    array_format=rr.Box2DFormat.XYXY,
                    class_ids=class_id,
                ),
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

    try:
        hocap_labels: HocapSequence = exoego_sequence.labels  # type: ignore[assignment]
        betas: Float32[ndarray, "10"] = hocap_labels.mano_stack.betas.astype(np.float32, copy=False)
    except Exception as e:
        betas: Float32[ndarray, "10"] = np.zeros((10,), dtype=np.float32)
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

        log_mano_outputs(
            hand_state=hand_state,
            tracker=mv_hand_tracker,
            parent_log_path=parent_log_path,
            pinhole_param_list=exo_sequence.exo_cam_list,
            recording=config.rr_config.rec_stream,
        )

    print(f"Total time taken: {timer() - start_time:.2f} seconds")
