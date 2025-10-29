from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from timeit import default_timer as timer
from typing import Any, cast

import cv2
import numpy as np
import open3d as o3d
import rerun as rr
import rerun.blueprint as rrb
import torch
from jaxtyping import Bool, Float, Float32, Int, UInt8
from monopriors.apis.multiview_calibration import MultiViewCalibrator, MultiViewCalibratorConfig, MVCalibResults
from numpy import ndarray
from simplecv.apis.view_exoego import (
    LogPaths,
    SceneSetupResult,
    filter_out_of_bounds_keypoints,
    setup_scene,
)
from simplecv.camera_parameters import Intrinsics, PinholeParameters
from simplecv.configs.exoego_dataset_configs import AnnotatedExoEgoDatasetUnion
from simplecv.data.ego.base_ego import BaseEgoSequence
from simplecv.data.exo.base_exo import BaseExoSequence
from simplecv.data.exoego.base_exoego import BaseExoEgoSequence
from simplecv.data.skeleton.coco_133 import (
    COCO_133_ID2NAME,
    COCO_133_IDS,
    COCO_133_LINKS,
    FACE_IDX,
    LEFT_HAND_IDX,
    RIGHT_HAND_IDX,
)
from simplecv.ops.pc_utils import estimate_voxel_size
from simplecv.ops.triangulate import proj_3d_vectorized
from simplecv.ops.tsdf_depth_fuser import Open3DScaleInvariantFuser
from simplecv.rerun_custom_types import Points2DWithConfidence, Points3DWithConfidence, confidence_scores_to_rgb
from simplecv.rerun_log_utils import RerunTyroConfig, log_pinhole
from simplecv.video_io import MultiVideoReader
from tqdm import tqdm

from annotation_example.coco133_layers import (
    COCO133_LAYER_COLORS,
    COCO133_LAYER_LABELS,
    COCO133_PREDICTION_LAYER_TO_PATH,
    Coco133AnnotationLayer,
)
from annotation_example.hand_keypoints import FinalWilorPred, HandKeypointDetectorConfig, WilorHandKeypointDetector
from annotation_example.multiview_pose_estimator import MultiviewBodyTracker, MultiviewBodyTrackerConfig, MVHistory

np.set_printoptions(suppress=True)

SUPPORTED_IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg")
device = "cuda" if torch.cuda.is_available() else "cpu"


def create_view_container(
    *,
    ego_video_log_paths: list[Path] | None = None,
    exo_video_log_paths: list[Path] | None = None,
    max_ego_videos_to_log: int = 4,
    max_exo_videos_to_log: int = 8,
) -> rrb.ContainerLike:
    """Create a Rerun blueprint for visualizing ego- and exo-centric streams.

    Args:
        ego_video_log_paths (list[Path] | None): Optional set of ego video entity
            roots; each path becomes a tabbed 2D view alongside the spatial view.
        exo_video_log_paths (list[Path] | None): Optional set of exo video entity
            roots; each path becomes a tabbed 2D view beneath the spatial view.
        max_exo_videos_to_log (Literal[4, 8]): Maximum number of exo video panels
            to materialize when ``exo_video_log_paths`` is provided.
        max_ego_videos_to_log (Literal[4, 8]): Maximum number of ego video panels
            to materialize when ``ego_video_log_paths`` is provided.

    Returns:
        rrb.Blueprint: Assembled layout containing the configured views.
    """
    main_view = rrb.Spatial3DView(
        origin="/",
        contents=[
            "+ $origin/**",
            "- /world/gt/env_pointcloud",  # hide raw point clouds by default
        ],
        line_grid=rrb.archetypes.LineGrid3D(visible=False),
    )

    if ego_video_log_paths is not None:
        ego_view = rrb.Vertical(
            contents=[
                rrb.Tabs(
                    rrb.Spatial2DView(origin=f"{video_log_path.parent}"),
                )
                for video_log_path in ego_video_log_paths[:max_ego_videos_to_log]
            ]
        )
        main_view = rrb.Horizontal(
            contents=[main_view, ego_view],
            column_shares=[4, 1],
        )

    if exo_video_log_paths is not None:
        exo_view = rrb.Horizontal(
            contents=[
                rrb.Tabs(
                    rrb.Spatial2DView(origin=f"{video_log_path.parent}"),
                )
                for video_log_path in exo_video_log_paths[:max_exo_videos_to_log]
            ]
        )
        main_view = rrb.Vertical(
            contents=[main_view, exo_view],
            row_shares=[4, 1],
        )

    final_view = rrb.Horizontal(
        contents=[main_view],
        column_shares=[4, 1],
    )

    return final_view


def compute_square_bbox(
    hand_uv: Float32[ndarray, "21 2"],
    *,
    intrinsics: Intrinsics,
    expansion_ratio: float,
) -> Float32[ndarray, "4"] | None:
    """Return an expanded square XYXY bbox from finite 2D keypoints or ``None`` if invalid."""
    if not np.isfinite(hand_uv).all():
        return None

    min_xy: Float32[ndarray, "2"] = np.nanmin(hand_uv, axis=0).astype(np.float32, copy=False)
    max_xy: Float32[ndarray, "2"] = np.nanmax(hand_uv, axis=0).astype(np.float32, copy=False)

    side_length: float = float(max(max_xy[0] - min_xy[0], max_xy[1] - min_xy[1]))
    if not np.isfinite(side_length) or side_length <= 0.0:
        return None

    center_xy: Float32[ndarray, "2"] = ((min_xy + max_xy) / np.float32(2.0)).astype(np.float32, copy=False)
    half_side: float = 0.5 * side_length * (1.0 + expansion_ratio)
    if half_side <= 0.0:
        return None

    x1: float = float(center_xy[0] - half_side)
    y1: float = float(center_xy[1] - half_side)
    x2: float = float(center_xy[0] + half_side)
    y2: float = float(center_xy[1] + half_side)

    if intrinsics.width is not None:
        width_limit: float = float(intrinsics.width - 1)
        x1 = float(np.clip(x1, 0.0, width_limit))
        x2 = float(np.clip(x2, 0.0, width_limit))
    if intrinsics.height is not None:
        height_limit: float = float(intrinsics.height - 1)
        y1 = float(np.clip(y1, 0.0, height_limit))
        y2 = float(np.clip(y2, 0.0, height_limit))

    if x2 <= x1 or y2 <= y1:
        return None

    xyxy: Float32[ndarray, "4"] = np.array([x1, y1, x2, y2], dtype=np.float32)
    return xyxy


def set_annotation_context(*, recording: rr.RecordingStream | None) -> None:
    keypoint_infos: list[rr.AnnotationInfo] = [
        rr.AnnotationInfo(id=id, label=name) for id, name in COCO_133_ID2NAME.items()
    ]
    class_descriptions: list[rr.ClassDescription] = []
    for layer in (
        Coco133AnnotationLayer.GT,
        Coco133AnnotationLayer.RAW_2D,
        Coco133AnnotationLayer.TRACKED_2D,
        Coco133AnnotationLayer.PROJECTED_2D,
        Coco133AnnotationLayer.OPTIMIZED_2D,
    ):
        label: str = COCO133_LAYER_LABELS[layer]
        color: tuple[int, int, int] = COCO133_LAYER_COLORS[layer]
        class_descriptions.append(
            rr.ClassDescription(
                info=rr.AnnotationInfo(id=int(layer), label=label, color=color),
                keypoint_annotations=keypoint_infos,
                keypoint_connections=COCO_133_LINKS,
            )
        )
    rr.log(
        "/",
        rr.AnnotationContext(class_descriptions),
        static=True,
        recording=recording,
    )


def timestamp_to_frame_index(time_ns: int, frame_timestamps_ns: Int[ndarray, "num_frames"]) -> int:
    """Map a timestamp (ns) to the closest frame idx at-or-before that time.

    The mapping mirrors how VideoFrameReference columns are generated from
    AssetVideo.read_frame_timestamps_nanos().
    """

    idx: int = int(np.searchsorted(frame_timestamps_ns, time_ns, side="right") - 1)
    return max(0, min(idx, len(frame_timestamps_ns) - 1))


def frame_index_to_timestamp(frame_timestamps_ns: Int[ndarray, "num_frames"], frame_index: int) -> int:
    """Return the nanosecond timestamp associated with a frame index."""
    if frame_index < 0 or frame_index >= int(frame_timestamps_ns.shape[0]):
        msg = f"frame_index {frame_index} is outside the valid range [0, {frame_timestamps_ns.shape[0] - 1}]"
        raise IndexError(msg)
    timestamp_ns: int = int(frame_timestamps_ns[frame_index])
    return timestamp_ns


def predict_kpts3d_from_calibrated_videos(
    exo_video_readers: MultiVideoReader,
    exo_cam_list: list[PinholeParameters],
    pose_tracker: MultiviewBodyTracker,
    top_half_mask: Bool[ndarray, "_"],
    shortest_timestamp: Int[ndarray, "num_frames"],
    parent_log_path: Path,
    max_frames: int | None = None,
    recording: rr.RecordingStream | None = None,
) -> list[Float32[ndarray, "n_kpts 4"]]:
    Pall: Float32[ndarray, "n_views 3 4"] = np.stack([cam.projection_matrix for cam in exo_cam_list]).astype(np.float32)
    exo_frame_timestamps_list: list[Int[ndarray, "num_frames"]] = [
        rr.AssetVideo(path=video_path).read_frame_timestamps_nanos() for video_path in exo_video_readers.video_paths
    ]

    pbar: object = tqdm(
        shortest_timestamp,
        total=len(shortest_timestamp) if max_frames is None else min(len(shortest_timestamp), max_frames),
    )
    pbar_iter: Iterable[int] = cast(Iterable[int], pbar)
    conf_thresh: float = pose_tracker.config.keypoint_threshold
    mv_output: MVHistory = MVHistory()
    xyzc_list: list[Float32[ndarray, "n_kpts 4"]] = []
    gt_class_id: int = int(Coco133AnnotationLayer.GT)
    for ts_idx, timestamp in enumerate(pbar_iter):
        if max_frames is not None and ts_idx >= max_frames:
            break
        # Anchor logging on a shared nanosecond timeline so 24 fps ego clips and 30 fps exo clips stay synchronized in the viewer.
        rr.set_time(timeline="video_time", duration=np.timedelta64(int(timestamp), "ns"))
        # Convert the unified timestamp into per-camera frame indices to absorb fps drift across exo recordings.
        frame_indices: list[int] = [
            timestamp_to_frame_index(time_ns=int(timestamp), frame_timestamps_ns=frame_timestamps)
            for frame_timestamps in exo_frame_timestamps_list
        ]
        bgr_list: list[UInt8[ndarray, "H W 3"]] = []
        for video_reader, frame_idx in zip(exo_video_readers.video_readers, frame_indices, strict=True):
            frame_raw: object = video_reader[frame_idx]
            if frame_raw is None:
                raise ValueError(f"Missing frame for index {frame_idx} in multi-view reader.")
            frame_array: UInt8[ndarray, "H W 3"] = np.asarray(frame_raw, dtype=np.uint8)
            if frame_array.ndim != 3 or frame_array.shape[2] != 3:
                raise ValueError(f"Expected frame with shape (*, *, 3), got {frame_array.shape} for index {frame_idx}.")
            bgr_list.append(frame_array)
        mv_output: MVHistory = pose_tracker(
            bgr_list=bgr_list,
            pinhole_list=exo_cam_list,
            pred_state=mv_output,
            recording=recording,
        )

        xyzc_list.append(mv_output.xyzc_t if mv_output.xyzc_t is not None else np.full((133, 4), np.nan))
        # send 3d keypoints
        if mv_output.xyzc_t is None:
            continue

        vis_xyz: Float32[ndarray, "n_kpts 3"] = mv_output.xyzc_t[:, :3].copy()
        vis_scores_3d: Float32[ndarray, "n_kpts"] = mv_output.xyzc_t[:, 3].copy()  # noqa: UP037
        # filter to only include the desired keypoints
        vis_xyz[~top_half_mask, :] = np.nan
        vis_scores_3d[~top_half_mask] = np.nan
        # filter out low-confidence keypoints
        vis_xyz[vis_scores_3d < conf_thresh, :] = np.nan
        vis_scores_3d[vis_scores_3d < conf_thresh] = np.nan
        # get hands idx and check their average confidence
        left_hand_conf = vis_scores_3d[LEFT_HAND_IDX].mean()
        right_hand_conf = vis_scores_3d[RIGHT_HAND_IDX].mean()
        # if either hand is below 0.6 confidence, remove all hand keypoints
        if left_hand_conf < conf_thresh:
            vis_xyz[LEFT_HAND_IDX, :] = np.nan
            vis_scores_3d[LEFT_HAND_IDX] = np.nan
        if right_hand_conf < conf_thresh:
            vis_xyz[RIGHT_HAND_IDX, :] = np.nan
            vis_scores_3d[RIGHT_HAND_IDX] = np.nan

        confidence_rgb_stack: UInt8[ndarray, "1 n_kpts 3"] = confidence_scores_to_rgb(
            vis_scores_3d[np.newaxis, :, np.newaxis]
        )
        confidence_rgb: UInt8[ndarray, "n_kpts 3"] = confidence_rgb_stack[0]

        rr.log(
            str(parent_log_path / "gt" / "coco133_xyz"),
            Points3DWithConfidence(
                positions=vis_xyz,
                confidences=vis_scores_3d,
                class_ids=gt_class_id,
                keypoint_ids=COCO_133_IDS,
                show_labels=False,
                colors=confidence_rgb,
            ),
            recording=recording,
        )
        # project 3d keypoints into 2d and log
        xyz_hom: Float32[ndarray, "n_kpts 4"] = np.concatenate(
            [vis_xyz, np.ones((vis_xyz.shape[0], 1), dtype=np.float32)], axis=1
        )
        xyz_hom_stack: Float32[ndarray, "1 n_kpts 4"] = np.stack([xyz_hom], axis=0)
        uv_exo_stack: Float[ndarray, "1 n_views 133 2"] = proj_3d_vectorized(xyz_hom=xyz_hom_stack, P=Pall)
        uv_exo: Float32[ndarray, "n_views 133 2"] = uv_exo_stack[0]
        for uv_view, exo_cam in zip(uv_exo, exo_cam_list, strict=True):
            uv: Float32[ndarray, "133 2"] = uv_view.astype(np.float32, copy=True)
            confidences_view: Float32[ndarray, "133"] = vis_scores_3d.astype(np.float32, copy=True)

            pinhole_log_path = parent_log_path / "exo" / exo_cam.name / "pinhole"
            confidence_rgb_view_stack: UInt8[ndarray, "1 n_kpts 3"] = confidence_scores_to_rgb(
                confidences_view[np.newaxis, :, np.newaxis]
            )
            confidence_rgb_view: UInt8[ndarray, "n_kpts 3"] = confidence_rgb_view_stack[0]
            rr.log(
                str(pinhole_log_path / "gt" / "coco133_uv"),
                Points2DWithConfidence(
                    positions=uv,
                    confidences=confidences_view,
                    class_ids=gt_class_id,
                    keypoint_ids=COCO_133_IDS,
                    show_labels=False,
                    colors=confidence_rgb_view,
                ),
                recording=recording,
            )
        # Mask out keypoints not in the top half to avoid visualizing irrelevant or missing data.
        mv_output.xyzc_t[~top_half_mask, :] = np.nan

    return xyzc_list


@dataclass
class RRDPipelineConfig:
    rr_config: RerunTyroConfig
    """Configuration for rerun logging."""
    dataset: AnnotatedExoEgoDatasetUnion
    """Dataset factory capable of producing an annotated ``BaseExoEgoSequence``."""
    calib_confg: MultiViewCalibratorConfig = field(default_factory=MultiViewCalibratorConfig)
    """Parameters forwarded to the multi-view calibrator."""
    tracker_config: MultiviewBodyTrackerConfig = field(default_factory=MultiviewBodyTrackerConfig)
    """Configuration for the multiview body tracker."""
    calib_ts_nano: int | None = None
    """Optional nanosecond timestamp used to select calibration frames for cameras and MANO."""
    max_frames: int | None = None
    """Maximum number of frames to process. If None, all frames are processed."""


def main(config: RRDPipelineConfig) -> None:
    """Preserve the Tyro CLI entry point while delegating to the reusable pipeline helper."""
    run_full_exoego_pipeline(config=config, recording=None)


def run_full_exoego_pipeline(config: RRDPipelineConfig, recording: rr.RecordingStream | None = None) -> None:
    """Execute the Exo/Ego pipeline; split from main so UI backends can supply explicit Rerun recordings."""
    parent_log_path = Path("world")
    timeline = "video_time"
    projected_variant: str = COCO133_PREDICTION_LAYER_TO_PATH[Coco133AnnotationLayer.PROJECTED_2D]
    projected_class_id: int = int(Coco133AnnotationLayer.PROJECTED_2D)

    ###################
    # 0. Parse inputs #
    ###################
    exoego_sequence: BaseExoEgoSequence = config.dataset.setup()  # one-liner
    rr.log("/", exoego_sequence.world_coordinate_system, static=True, recording=recording)
    set_annotation_context(recording=recording)

    parent_log_path = Path("world")
    timeline: str = "video_time"

    scene_setup_result: SceneSetupResult = setup_scene(exoego_sequence, parent_log_path, timeline, recording=recording)
    log_paths: LogPaths = scene_setup_result.log_paths
    shortest_timestamp: Int[ndarray, "n_frames"] = scene_setup_result.shortest_timestamp

    exo_video_log_paths_opt: list[Path] | None = log_paths.exo_video_log_paths
    if exo_video_log_paths_opt is None:
        raise ValueError("Scene setup must return exo video log paths.")
    ego_video_log_paths_opt: list[Path] | None = log_paths.ego_video_log_paths
    if ego_video_log_paths_opt is None:
        raise ValueError("Scene setup must return ego video log paths.")

    exo_video_log_paths: list[Path] = list(exo_video_log_paths_opt)
    ego_video_log_paths: list[Path] = list(ego_video_log_paths_opt)

    final_container: rrb.ContainerLike = create_view_container(
        exo_video_log_paths=exo_video_log_paths,
        ego_video_log_paths=ego_video_log_paths,
    )
    blueprint = rrb.Blueprint(
        final_container,
        collapse_panels=True,
    )
    # Blueprint routing still relies on global stream; callers providing a recording
    # should ensure it is set active before invoking this pipeline.
    rr.send_blueprint(blueprint, recording=recording)

    exo_sequence_obj: object = exoego_sequence.exo_sequence
    if exo_sequence_obj is None:
        raise ValueError("Dataset setup failed to provide an exo sequence.")
    ego_sequence_obj: object = exoego_sequence.ego_sequence
    if ego_sequence_obj is None:
        raise ValueError("Dataset setup failed to provide an ego sequence.")

    exo_sequence: BaseExoSequence = cast(BaseExoSequence, exo_sequence_obj)
    ego_sequence: BaseEgoSequence = cast(BaseEgoSequence, ego_sequence_obj)

    exo_mv_reader: MultiVideoReader = exo_sequence.exo_video_readers
    ego_mv_reader: MultiVideoReader = ego_sequence.ego_video_readers
    if shortest_timestamp.size == 0:
        raise ValueError("Scene setup returned no timestamps to derive calibration frame.")

    ##############################################
    # 1. Load calibration frames from all videos #
    ##############################################
    calib_timestamp_ns: int = int(shortest_timestamp[-1]) if config.calib_ts_nano is None else int(config.calib_ts_nano)
    ego_camera_names: list[str] = [path.parent.parent.name for path in ego_video_log_paths]
    ego_rgb_indices: list[int] = [
        idx for idx, camera_name in enumerate(ego_camera_names) if "rgb" in camera_name.lower()
    ]
    ego_rgb_index_set: set[int] = set(ego_rgb_indices)
    ego_rgb_video_log_paths: list[Path] = [ego_video_log_paths[idx] for idx in ego_rgb_indices]

    # load the exo frames first
    # Need to get ts list as exo and ego can have different fps and drift
    exo_frame_timestamp_list: list[Int[ndarray, "num_frames"]] = [
        rr.AssetVideo(path=video_path).read_frame_timestamps_nanos() for video_path in exo_mv_reader.video_paths
    ]
    exo_calib_indices: list[int] = [
        timestamp_to_frame_index(time_ns=calib_timestamp_ns, frame_timestamps_ns=frame_timestamps)
        for frame_timestamps in exo_frame_timestamp_list
    ]
    bgr_list_exo: list[UInt8[ndarray, "H W 3"]] = []
    for reader, frame_idx in zip(exo_mv_reader.video_readers, exo_calib_indices, strict=True):
        frame_obj: Any = reader[frame_idx]
        if frame_obj is None:
            raise ValueError(f"Missing exo frame at index {frame_idx}.")
        frame_array_untyped: np.ndarray = np.asarray(frame_obj, dtype=np.uint8)
        frame_array: UInt8[ndarray, "H W 3"] = frame_array_untyped
        if frame_array.ndim != 3 or frame_array.shape[2] != 3:
            raise ValueError(f"Expected BGR frame with shape (*, *, 3), got {frame_array.shape} at index {frame_idx}.")
        bgr_list_exo.append(frame_array)

    # ego frames next, but only the rgb cameras
    ego_frame_timestamp_list: list[Int[ndarray, "num_frames"]] = [
        rr.AssetVideo(path=video_path).read_frame_timestamps_nanos() for video_path in ego_mv_reader.video_paths
    ]
    ego_calib_indices: list[int] = [
        timestamp_to_frame_index(time_ns=calib_timestamp_ns, frame_timestamps_ns=frame_timestamps)
        for frame_timestamps in ego_frame_timestamp_list
    ]
    bgr_list_ego: list[UInt8[ndarray, "H W 3"]] = []
    if ego_rgb_indices:
        for camera_idx, reader in enumerate(ego_mv_reader.video_readers):
            if camera_idx not in ego_rgb_index_set:
                continue
            frame_idx: int = ego_calib_indices[camera_idx]
            frame_obj: Any = reader[frame_idx]
            if frame_obj is None:
                raise ValueError(f"Missing ego frame at index {frame_idx} for camera {camera_idx}.")
            frame_array_untyped: np.ndarray = np.asarray(frame_obj, dtype=np.uint8)
            frame_array: UInt8[ndarray, "H W 3"] = frame_array_untyped
            if frame_array.ndim != 3 or frame_array.shape[2] != 3:
                raise ValueError(
                    f"Expected BGR frame with shape (*, *, 3), got {frame_array.shape} at index {frame_idx}."
                )
            bgr_list_ego.append(frame_array)

    # convert all to rgb
    rgb_list_exo: list[UInt8[ndarray, "H W 3"]] = []
    for bgr in bgr_list_exo:
        rgb_frame: UInt8[ndarray, "H W 3"] = np.asarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), dtype=np.uint8)
        if rgb_frame.ndim != 3 or rgb_frame.shape[2] != 3:
            raise ValueError(f"Expected RGB frame with shape (*, *, 3), got {rgb_frame.shape}.")
        rgb_list_exo.append(rgb_frame)

    rgb_list_ego: list[UInt8[ndarray, "H W 3"]] = []
    for bgr in bgr_list_ego:
        rgb_frame: UInt8[ndarray, "H W 3"] = np.asarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), dtype=np.uint8)
        if rgb_frame.ndim != 3 or rgb_frame.shape[2] != 3:
            raise ValueError(f"Expected RGB frame with shape (*, *, 3), got {rgb_frame.shape}.")
        rgb_list_ego.append(rgb_frame)

    rgb_list: list[UInt8[ndarray, "H W 3"]] = [*rgb_list_exo, *rgb_list_ego]

    input_log_paths: list[Path] = [*exo_video_log_paths, *ego_rgb_video_log_paths]
    exo_ts: Int[ndarray, "num_frames"] = shortest_timestamp

    ############################
    # 2. Calibrate Exo Cameras #
    ############################
    start: float = timer()

    rr.set_time(timeline=timeline, duration=np.timedelta64(0, "ns"), recording=recording)
    hand_kpt_detector: WilorHandKeypointDetector = WilorHandKeypointDetector(
        cfg=HandKeypointDetectorConfig(verbose=False)
    )

    mv_calibrator: MultiViewCalibrator = MultiViewCalibrator(parent_log_path=parent_log_path, config=config.calib_confg)
    mv_calib_results: MVCalibResults = mv_calibrator(rgb_list=rgb_list)

    pinhole_param_list: list[PinholeParameters] = mv_calib_results.pinhole_param_list
    exo_pinhole_param_list: list[PinholeParameters] = pinhole_param_list[: len(rgb_list_exo)]
    ego_pinhole_param_list: list[PinholeParameters] = pinhole_param_list[len(rgb_list_exo) :]
    # replace cam names with those from the dataset for easier identification
    for cam, log_path in zip(exo_pinhole_param_list, exo_video_log_paths, strict=True):
        cam.name = log_path.parent.parent.name
    for cam, log_path in zip(ego_pinhole_param_list, ego_rgb_video_log_paths, strict=True):
        cam.name = log_path.parent.parent.name

    assert len(exo_pinhole_param_list) == len(rgb_list_exo)
    assert len(ego_pinhole_param_list) == len(rgb_list_ego)

    pcd: o3d.geometry.PointCloud = mv_calib_results.pcd
    # Automatically determine optimal voxel size based on point cloud characteristics
    voxel_size: float = estimate_voxel_size(np.asarray(pcd.points, dtype=np.float32), target_points=50_000)
    pcd_ds = pcd.voxel_down_sample(voxel_size)

    for pinhole, input_log_path in zip(pinhole_param_list, input_log_paths, strict=True):
        cam_log_path: Path = input_log_path.parent.parent
        log_pinhole(camera=pinhole, cam_log_path=cam_log_path, image_plane_distance=0.1, static=True)

    filtered_points: Float[ndarray, "final_points 3"] = np.asarray(pcd_ds.points, dtype=np.float32)
    filtered_colors: Float[ndarray, "final_points 3"] = np.asarray(pcd_ds.colors, dtype=np.float32)

    rr.log(
        str(parent_log_path / "gt" / "env_pointcloud"),
        rr.Points3D(
            filtered_points,
            colors=filtered_colors,
        ),
        static=True,
        recording=recording,
    )
    #####################################
    # 3. Fuse Depths into TSDF Mesh     #
    #####################################
    if mv_calib_results.depth_list and mv_calib_results.pinhole_param_list:
        depth_fuser = Open3DScaleInvariantFuser(grid_resolution=512)
        reference_points: Float32[ndarray, "num_points 3"] = np.asarray(pcd.points, dtype=np.float32)
        depth_fuser.initialise_from_points(reference_points)

        for depth_map, pinhole_param, rgb in zip(
            mv_calib_results.depth_list,
            mv_calib_results.pinhole_param_list,
            rgb_list,
            strict=True,
        ):
            depth_fuser.fuse_frame(depth_hw=depth_map, pinhole=pinhole_param, rgb_hw3=rgb)

        gt_mesh: o3d.geometry.TriangleMesh = depth_fuser.get_mesh()
        gt_mesh.compute_vertex_normals()

        vertex_positions: Float32[ndarray, "num_vertices 3"] = np.asarray(gt_mesh.vertices, dtype=np.float32)
        triangle_indices: Int[ndarray, "num_faces 3"] = np.asarray(gt_mesh.triangles, dtype=np.int32)

        vertex_normals: Float32[ndarray, "num_vertices 3"] = np.asarray(gt_mesh.vertex_normals, dtype=np.float32)
        vertex_colors: Float32[ndarray, "num_vertices 3"] = np.asarray(gt_mesh.vertex_colors, dtype=np.float32)

        rr.log(
            str(parent_log_path / "gt" / "env_mesh"),
            rr.Mesh3D(
                vertex_positions=vertex_positions,
                triangle_indices=triangle_indices,
                vertex_normals=vertex_normals,
                vertex_colors=vertex_colors,
            ),
            static=True,
            recording=recording,
        )

    if exo_mv_reader is not None and exo_ts is not None:
        #########################################################
        # 5. Predict exo keypoints from calibrated video frames #
        #########################################################
        upper_body_filter_idx: Int[ndarray, "_"] = np.array([5, 6, 7, 8, 9, 10])
        wb_upper_body_filter_idx: Int[ndarray, "_"] = np.concatenate(
            [upper_body_filter_idx, FACE_IDX, LEFT_HAND_IDX, RIGHT_HAND_IDX]
        )
        # Create a boolean mask for all rows
        top_half_mask: Bool[ndarray, "_"] = np.isin(np.arange(133), wb_upper_body_filter_idx)
        pose_tracker = MultiviewBodyTracker(config.tracker_config, filter_body_idxes=wb_upper_body_filter_idx)

        xyzc_list: list[Float32[ndarray, "n_kpts 4"]] = predict_kpts3d_from_calibrated_videos(
            exo_video_readers=exo_mv_reader,
            exo_cam_list=exo_pinhole_param_list,
            pose_tracker=pose_tracker,
            top_half_mask=top_half_mask,
            shortest_timestamp=exo_ts,
            parent_log_path=parent_log_path,
            max_frames=config.max_frames,
            recording=recording,
        )
        bbox_expansion_percentage: float = 0.25
        keypoint_threshold: float = pose_tracker.config.keypoint_threshold
        #########################################################
        # 6. Predict ego keypoints from calibrated video frames #
        #########################################################
        if len(xyzc_list) > 0 and len(ego_pinhole_param_list) > 0:
            Pall_ego: Float32[ndarray, "n_views 3 4"] = np.stack(
                [cam.projection_matrix for cam in ego_pinhole_param_list]
            ).astype(np.float32)

            for idx, xyzc in enumerate(xyzc_list):
                rr.set_time(timeline=timeline, duration=np.timedelta64(int(exo_ts[idx]), "ns"), recording=recording)
                bgr_list_ego: list[UInt8[ndarray, "H W 3"]] = []
                for camera_idx, frame in enumerate(ego_mv_reader[idx]):
                    if camera_idx not in ego_rgb_index_set:
                        continue
                    if frame is None:
                        raise ValueError(f"Missing ego frame at batch index {camera_idx} for timestamp index {idx}.")
                    frame_array: UInt8[ndarray, "H W 3"] = np.asarray(frame, dtype=np.uint8)
                    if frame_array.ndim != 3 or frame_array.shape[2] != 3:
                        raise ValueError(
                            f"Expected BGR frame with shape (*, *, 3), got {frame_array.shape} at index {camera_idx}."
                        )
                    bgr_list_ego.append(frame_array)

                if len(bgr_list_ego) != len(ego_pinhole_param_list):
                    msg = (
                        "Filtered ego frame count does not match calibrated ego cameras: "
                        f"{len(bgr_list_ego)} vs {len(ego_pinhole_param_list)}."
                    )
                    raise ValueError(msg)

                if xyzc is None:
                    continue
                vis_xyz: Float32[ndarray, "n_kpts=133 3"] = xyzc[:, :3].copy()
                vis_scores_3d: Float32[ndarray, "n_kpts=133"] = xyzc[:, 3].copy()  # noqa: UP037
                # filter to only include the desired keypoints
                vis_xyz[~top_half_mask, :] = np.nan
                vis_scores_3d[~top_half_mask] = np.nan
                # filter out low-confidence keypoints
                vis_xyz[vis_scores_3d < keypoint_threshold, :] = np.nan
                vis_scores_3d[vis_scores_3d < keypoint_threshold] = np.nan

                # project 3d keypoints into 2d
                xyz_hom: Float32[ndarray, "n_kpts=133 4"] = np.concatenate(
                    [vis_xyz, np.ones((vis_xyz.shape[0], 1), dtype=np.float32)], axis=1
                )
                xyz_hom_stack: Float32[ndarray, "n_frames=1 n_kpts=133 4"] = np.stack([xyz_hom], axis=0)
                uv_ego_stack: Float[ndarray, "n_frames=1 n_views n_kpts=133 2"] = proj_3d_vectorized(
                    xyz_hom=xyz_hom_stack, P=Pall_ego
                )
                uv_ego: Float32[ndarray, "n_views n_kpts=133 2"] = uv_ego_stack[0]

                for view_idx, (uv_view, ego_cam) in enumerate(zip(uv_ego, ego_pinhole_param_list, strict=True)):
                    pinhole_log_path: Path = parent_log_path / "ego" / ego_cam.name / "pinhole"
                    uv: Float32[ndarray, "n_kpts=133 2"] = uv_view.astype(np.float32, copy=True)
                    projection_matrix: Float32[ndarray, "3 4"] = ego_cam.projection_matrix.astype(
                        np.float32, copy=False
                    )
                    homog_cam: Float32[ndarray, "n_kpts=133 3"] = np.einsum("ij,nj->ni", projection_matrix, xyz_hom)
                    depth_cam: Float32[ndarray, "n_kpts=133"] = homog_cam[:, 2]
                    # Reject keypoints with non-positive homogeneous depth (behind the camera).
                    invalid_depth_mask: Bool[ndarray, "n_kpts=133"] = np.logical_or(
                        ~np.isfinite(depth_cam),
                        depth_cam <= 0.0,
                    )
                    uv[invalid_depth_mask, :] = np.nan
                    # filter out keypoints that are out of bounds
                    uv = filter_out_of_bounds_keypoints(uv, ego_cam, margin_percentage=0.0)
                    invalid_uv_mask: Bool[ndarray, "n_kpts=133"] = np.any(~np.isfinite(uv), axis=1)

                    confidences_view: Float32[ndarray, "n_kpts=133"] = vis_scores_3d.astype(np.float32, copy=True)
                    confidences_view[invalid_uv_mask] = np.nan
                    rgb_hw3: UInt8[ndarray, "H W 3"] = bgr_list_ego[view_idx][..., ::-1]

                    left_bbox_infer: Float32[ndarray, "4"] | None = compute_square_bbox(
                        uv[LEFT_HAND_IDX, :],
                        intrinsics=ego_cam.intrinsics,
                        expansion_ratio=bbox_expansion_percentage,
                    )
                    if left_bbox_infer is not None:
                        xyxy_left: Float32[ndarray, "1 4"] = left_bbox_infer[np.newaxis, :]
                        wilor_left: FinalWilorPred = hand_kpt_detector(
                            rgb_hw3=rgb_hw3,
                            xyxy=xyxy_left,
                            handedness="left",
                        )
                        left_uv_pred: Float32[ndarray, "1 21 2"] = wilor_left.pred_keypoints_2d.astype(
                            np.float32, copy=False
                        )
                        uv[LEFT_HAND_IDX, :] = left_uv_pred[0]

                    right_bbox_infer: Float32[ndarray, "4"] | None = compute_square_bbox(
                        uv[RIGHT_HAND_IDX, :],
                        intrinsics=ego_cam.intrinsics,
                        expansion_ratio=bbox_expansion_percentage,
                    )
                    if right_bbox_infer is not None:
                        xyxy_right: Float32[ndarray, "1 4"] = right_bbox_infer[np.newaxis, :]
                        wilor_right: FinalWilorPred = hand_kpt_detector(
                            rgb_hw3=rgb_hw3,
                            xyxy=xyxy_right,
                            handedness="right",
                        )
                        right_uv_pred: Float32[ndarray, "1 21 2"] = wilor_right.pred_keypoints_2d.astype(
                            np.float32, copy=False
                        )
                        uv[RIGHT_HAND_IDX, :] = right_uv_pred[0]

                    left_bbox_log: Float32[ndarray, "4"] | None = compute_square_bbox(
                        uv[LEFT_HAND_IDX, :],
                        intrinsics=ego_cam.intrinsics,
                        expansion_ratio=bbox_expansion_percentage,
                    )
                    if left_bbox_log is not None:
                        lh_xyxy: Float32[ndarray, "4"] = left_bbox_log.astype(np.float32, copy=False)
                        rr.log(
                            f"{pinhole_log_path}/video/left_hand_bbox",
                            rr.Boxes2D(array=lh_xyxy, array_format=rr.Box2DFormat.XYXY),
                            recording=recording,
                        )
                    else:
                        rr.log(
                            f"{pinhole_log_path}/video/left_hand_bbox",
                            rr.Clear(recursive=True),
                            recording=recording,
                        )

                    right_bbox_log: Float32[ndarray, "4"] | None = compute_square_bbox(
                        uv[RIGHT_HAND_IDX, :],
                        intrinsics=ego_cam.intrinsics,
                        expansion_ratio=bbox_expansion_percentage,
                    )
                    if right_bbox_log is not None:
                        rh_xyxy: Float32[ndarray, "4"] = right_bbox_log.astype(np.float32, copy=False)
                        rr.log(
                            f"{pinhole_log_path}/video/right_hand_bbox",
                            rr.Boxes2D(array=rh_xyxy, array_format=rr.Box2DFormat.XYXY),
                            recording=recording,
                        )
                    else:
                        rr.log(
                            f"{pinhole_log_path}/video/right_hand_bbox",
                            rr.Clear(recursive=True),
                            recording=recording,
                        )

                    confidence_rgb_view_stack: UInt8[ndarray, "1 n_kpts 3"] = confidence_scores_to_rgb(
                        confidences_view[np.newaxis, :, np.newaxis]
                    )
                    confidence_rgb_view: UInt8[ndarray, "n_kpts 3"] = confidence_rgb_view_stack[0]
                    rr.log(
                        str(pinhole_log_path / "pred" / "coco133_uv" / projected_variant),
                        Points2DWithConfidence(
                            positions=uv,
                            confidences=confidences_view,
                            class_ids=projected_class_id,
                            keypoint_ids=COCO_133_IDS,
                            show_labels=False,
                            colors=confidence_rgb_view,
                        ),
                        recording=recording,
                    )

    print(f"Inference completed in {timer() - start:.2f} seconds")
