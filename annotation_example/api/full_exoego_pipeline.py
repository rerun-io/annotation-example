from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from timeit import default_timer as timer
from typing import Literal, NamedTuple, cast

import cv2
import numpy as np
import open3d as o3d
import rerun as rr
import rerun.blueprint as rrb
import torch
from jaxtyping import Float, Float32, Int, UInt8
from monopriors.apis.multiview_calibration import MultiViewCalibrator, MultiViewCalibratorConfig, MVCalibResults
from numpy import ndarray
from simplecv.apis.view_exoego import (
    LogPaths,
    SceneSetupResult,
    create_blueprint,
    filter_out_of_bounds_keypoints,
    setup_scene,
)
from simplecv.camera_parameters import Intrinsics, PinholeParameters
from simplecv.configs.exoego_dataset_configs import AnnotatedExoEgoDatasetUnion
from simplecv.data.exoego.base_exoego import BaseExoEgoSequence
from simplecv.data.skeleton.coco_133 import COCO_133_ID2NAME, COCO_133_IDS, COCO_133_LINKS
from simplecv.ops.pc_utils import estimate_voxel_size
from simplecv.ops.triangulate import proj_3d_vectorized
from simplecv.ops.tsdf_depth_fuser import Open3DScaleInvariantFuser
from simplecv.rerun_custom_types import Points2DWithConfidence, Points3DWithConfidence, confidence_scores_to_rgb
from simplecv.rerun_log_utils import RerunTyroConfig, log_pinhole, log_video
from simplecv.video_io import MultiVideoReader
from tqdm import tqdm
from wilor_nano.hand_keypoints import FinalWilorPred, WilorHandKeypointDetector

np.set_printoptions(suppress=True)

SUPPORTED_IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg")
device = "cuda" if torch.cuda.is_available() else "cpu"


def create_depth_views(parent_log_path: Path, camera_index: int) -> rrb.Tabs:
    """
    Create depth visualization tabs for a specific camera.

    Args:
        parent_log_path: Parent log path for the camera views
        camera_index: Index of the camera to create depth views for

    Returns:
        Tabs blueprint containing depth and filtered depth views
    """
    depth_views: rrb.Tabs = rrb.Tabs(
        contents=[
            rrb.Spatial2DView(
                origin=f"{parent_log_path}/exo/camera_{camera_index}/pinhole/depth",
                contents=[
                    "+ $origin/**",
                ],
                name="Depth",
            ),
            rrb.Spatial2DView(
                origin=f"{parent_log_path}/exo/camera_{camera_index}/pinhole/filtered_depth",
                contents=[
                    "+ $origin/**",
                ],
                name="Filtered Depth",
            ),
            rrb.Spatial2DView(
                origin=f"{parent_log_path}/camera_{camera_index}/pinhole/refined_depth",
                contents=[
                    "+ $origin/**",
                ],
                name="MoGe Depth",
            ),
        ],
        active_tab=2,
    )
    return depth_views


def create_camera_row(parent_log_path: Path, camera_index: int) -> rrb.Horizontal:
    """
    Create a single camera row with 3 views: content, depth, and confidence.

    Args:
        parent_log_path: Parent log path for the camera views
        camera_index: Index of the camera to create views for

    Returns:
        Horizontal blueprint containing pinhole content, depth views, and confidence map
    """
    camera_row: rrb.Horizontal = rrb.Horizontal(
        contents=[
            rrb.Spatial2DView(
                origin=f"{parent_log_path}/exo/camera_{camera_index}/pinhole/image",
                contents=[
                    "+ $origin/**",
                ],
                name="Image Content",
            ),
            create_depth_views(parent_log_path, camera_index),
            rrb.Spatial2DView(
                origin=f"{parent_log_path}/exo/camera_{camera_index}/pinhole/confidence",
                contents=[
                    "+ $origin/**",
                ],
                name="Confidence Map",
            ),
        ]
    )
    return camera_row


def chunk_cameras(num_cameras: int, chunk_size: int = 4) -> list[range]:
    """
    Group cameras into chunks of specified size.

    Args:
        num_cameras: Total number of cameras
        chunk_size: Maximum cameras per chunk (default 4)

    Returns:
        List of ranges representing camera chunks
    """
    chunks: list[range] = [range(i, min(i + chunk_size, num_cameras)) for i in range(0, num_cameras, chunk_size)]
    return chunks


def create_tabbed_camera_view(parent_log_path: Path, num_cameras: int) -> rrb.Tabs:
    """
    Create tabbed interface grouping cameras by 4s.

    Args:
        parent_log_path: Parent log path for the camera views
        num_cameras: Total number of cameras to display

    Returns:
        Tabs blueprint with each tab containing up to 4 camera rows
    """
    camera_chunks: list[range] = chunk_cameras(num_cameras)

    tabs: list[rrb.Vertical] = []
    for camera_range in camera_chunks:
        # Create camera rows for this chunk
        camera_rows: list[rrb.Horizontal] = [create_camera_row(parent_log_path, i) for i in camera_range]

        # Create tab name
        if camera_range.start + 1 == camera_range.stop:
            tab_name: str = f"Camera {camera_range.start + 1}"
        else:
            tab_name = f"Cameras {camera_range.start + 1}-{camera_range.stop}"

        # Create tab content
        tab_content: rrb.Vertical = rrb.Vertical(contents=camera_rows, name=tab_name)
        tabs.append(tab_content)

    tabbed_view: rrb.Tabs = rrb.Tabs(contents=tabs, name="Depths Tab")
    return tabbed_view


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


def create_blueprint_old(parent_log_path: Path, num_images: int, show_videos: bool = False) -> rrb.Blueprint:
    view3d = rrb.Spatial3DView(
        origin=f"{parent_log_path}",
        contents=[
            "+ $origin/**",
            f"- /{parent_log_path}/pointcloud",
            # don't include depths in the 3D view, as they can be very noisy
            *[f"- /{parent_log_path}/exo/camera_{i}/pinhole/depth" for i in range(num_images)],
            *[f"- /{parent_log_path}/exo/camera_{i}/pinhole/filtered_depth" for i in range(num_images)],
            *[f"- /{parent_log_path}/exo/camera_{i}/pinhole/refined_depth" for i in range(num_images)],
            *[f"- /{parent_log_path}/exo/camera_{i}/pinhole/confidence" for i in range(num_images)],
            *[f"- /{parent_log_path}/exo/camera_{i}/pinhole/image" for i in range(num_images)],
        ],
        line_grid=rrb.archetypes.LineGrid3D(visible=False),
    )

    # Create tabbed view that supports any number of cameras
    view_2d: rrb.Tabs = create_tabbed_camera_view(parent_log_path, num_images)
    if show_videos:
        view_2d_videos: rrb.Grid = rrb.Grid(
            contents=[
                rrb.Spatial2DView(origin=f"{parent_log_path}/exo/camera_{i}/pinhole/video", name=f"Video {i + 1}")
                for i in range(num_images)
            ],
            name="Videos Tab",
        )
        view_2d = rrb.Tabs(view_2d, view_2d_videos)

    blueprint = rrb.Blueprint(rrb.Horizontal(contents=[view3d, view_2d], column_shares=[3, 2]), collapse_panels=True)
    return blueprint


def set_annotation_context() -> None:
    rr.log(
        "/",
        rr.AnnotationContext(
            [
                rr.ClassDescription(
                    info=rr.AnnotationInfo(id=0, label="Coco Wholebody", color=(0, 0, 255)),
                    keypoint_annotations=[
                        rr.AnnotationInfo(id=id, label=name) for id, name in COCO_133_ID2NAME.items()
                    ],
                    keypoint_connections=COCO_133_LINKS,
                ),
            ]
        ),
        static=True,
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
    shortest_timestamp: Int[ndarray, "num_frames"],
    parent_log_path: Path,
    hand_kpt_detector: WilorHandKeypointDetector,
    max_frames: int | None = None,
) -> list[Float32[ndarray, "num_kpts 4"]]:
    upper_body_filter_idx = np.array([5, 6, 7, 8, 9, 10])
    face_idx = np.arange(23, 91)
    left_hand_idx = np.arange(91, 112)
    right_hand_idx = np.arange(112, 133)
    wb_upper_body_filter_idx = np.concatenate([upper_body_filter_idx, face_idx, left_hand_idx, right_hand_idx])

    # Create a boolean mask for all rows
    top_half_mask = np.isin(np.arange(133), wb_upper_body_filter_idx)
    bbox_expansion_ratio: float = 0.2

    pose_tracker = MultiviewBodyTracker(
        MultiviewBodyTrackerConfig(
            mode="wholebody",
            backend="onnxruntime",
            device="cuda",
            filter_body_idxes=wb_upper_body_filter_idx,
            cams_for_detection_idx=None,  # use all cameras
            perform_tracking=True,
            use_wilor=True,
        )
    )

    Pall: Float32[ndarray, "n_views 3 4"] = np.stack([cam.projection_matrix for cam in exo_cam_list]).astype(np.float32)
    exo_frame_timestamps_list: list[Int[ndarray, "num_frames"]] = [
        rr.AssetVideo(path=video_path).read_frame_timestamps_nanos() for video_path in exo_video_readers.video_paths
    ]

    pbar = tqdm(
        shortest_timestamp,
        total=len(shortest_timestamp) if max_frames is None else min(len(shortest_timestamp), max_frames),
    )
    conf_thresh: float = 0.7
    mv_output: MVHistory = MVHistory()
    pbar_iter: Iterable[int] = cast(Iterable[int], pbar)
    xyzc_list: list[Float32[ndarray, "num_kpts 4"]] = []
    for ts_idx, timestamp in enumerate(pbar_iter):
        if max_frames is not None and ts_idx >= max_frames:
            break
        rr.set_time(timeline="video_time", duration=np.timedelta64(int(timestamp), "ns"))
        frame_indices: list[int] = [
            timestamp_to_frame_index(time_ns=int(timestamp), frame_timestamps_ns=frame_timestamps)
            for frame_timestamps in exo_frame_timestamps_list
        ]
        bgr_list: list[UInt8[ndarray, "H W 3"]] = [
            video_reader[frame_idx]
            for video_reader, frame_idx in zip(exo_video_readers.video_readers, frame_indices, strict=True)
        ]
        mv_output: MVHistory = pose_tracker(
            bgr_list=bgr_list,
            pinhole_list=exo_cam_list,
            pred_state=mv_output,
            recording=None,
        )

        xyzc_list.append(mv_output.xyzc_t if mv_output.xyzc_t is not None else np.full((133, 4), np.nan))
        # send 3d keypoints
        if mv_output.xyzc_t is None:
            continue

        vis_xyz: Float32[ndarray, "num_kpts 3"] = mv_output.xyzc_t[:, :3].copy()
        vis_scores_3d: Float32[ndarray, "num_kpts"] = mv_output.xyzc_t[:, 3].copy()  # noqa: UP037
        # filter to only include the desired keypoints
        vis_xyz[~top_half_mask, :] = np.nan
        vis_scores_3d[~top_half_mask] = np.nan
        # filter out low-confidence keypoints
        vis_xyz[vis_scores_3d < conf_thresh, :] = np.nan
        vis_scores_3d[vis_scores_3d < conf_thresh] = np.nan
        # get hands idx and check their average confidence
        left_hand_conf = vis_scores_3d[left_hand_idx].mean()
        right_hand_conf = vis_scores_3d[right_hand_idx].mean()
        # if either hand is below 0.6 confidence, remove all hand keypoints
        if left_hand_conf < conf_thresh:
            vis_xyz[left_hand_idx, :] = np.nan
            vis_scores_3d[left_hand_idx] = np.nan
        if right_hand_conf < conf_thresh:
            vis_xyz[right_hand_idx, :] = np.nan
            vis_scores_3d[right_hand_idx] = np.nan

        confidence_rgb_stack: UInt8[ndarray, "1 num_kpts 3"] = confidence_scores_to_rgb(
            vis_scores_3d[np.newaxis, :, np.newaxis]
        )
        confidence_rgb: UInt8[ndarray, "num_kpts 3"] = confidence_rgb_stack[0]

        rr.log(
            f"{parent_log_path}/wholebody",
            Points3DWithConfidence(
                positions=vis_xyz,
                confidences=vis_scores_3d,
                class_ids=0,
                keypoint_ids=COCO_133_IDS,
                show_labels=False,
                colors=confidence_rgb,
            ),
            recording=None,
        )
        # project 3d keypoints into 2d and log
        xyz_hom: Float32[ndarray, "num_kpts 4"] = np.concatenate(
            [vis_xyz, np.ones((vis_xyz.shape[0], 1), dtype=np.float32)], axis=1
        )
        xyz_hom_stack: Float32[ndarray, "1 num_kpts 4"] = np.stack([xyz_hom], axis=0)
        uv_exo_stack: Float[ndarray, "1 n_views 133 2"] = proj_3d_vectorized(xyz_hom=xyz_hom_stack, P=Pall)
        uv_exo: Float32[ndarray, "n_views 133 2"] = uv_exo_stack[0]
        for view_idx, (uv_view, exo_cam) in enumerate(zip(uv_exo, exo_cam_list, strict=True)):
            uv: Float32[ndarray, "133 2"] = uv_view.astype(np.float32, copy=True)
            confidences_view: Float32[ndarray, "133"] = vis_scores_3d.astype(np.float32, copy=True)

            left_hand_uv: Float32[ndarray, "21 2"] = uv[left_hand_idx, :]
            right_hand_uv: Float32[ndarray, "21 2"] = uv[right_hand_idx, :]

            rgb_hw3: UInt8[ndarray, "H W 3"] = bgr_list[view_idx][..., ::-1]

            left_bbox: Float32[ndarray, "4"] | None = compute_square_bbox(
                left_hand_uv,
                intrinsics=exo_cam.intrinsics,
                expansion_ratio=bbox_expansion_ratio,
            )
            if left_bbox is not None:
                xyxy_left: Float32[ndarray, "1 4"] = left_bbox[np.newaxis, :]
                wilor_left: FinalWilorPred = hand_kpt_detector(
                    rgb_hw3=rgb_hw3,
                    xyxy=xyxy_left,
                    handedness="left",
                )
                left_uv_pred: Float32[ndarray, "1 21 2"] = wilor_left.pred_keypoints_2d.astype(np.float32, copy=False)
                uv[left_hand_idx, :] = left_uv_pred[0]
                left_conf_pred: Float32[ndarray, "1 21"] = wilor_left.confidence_2d.astype(np.float32, copy=False)
                confidences_view[left_hand_idx] = left_conf_pred[0]

            right_bbox: Float32[ndarray, "4"] | None = compute_square_bbox(
                right_hand_uv,
                intrinsics=exo_cam.intrinsics,
                expansion_ratio=bbox_expansion_ratio,
            )
            if right_bbox is not None:
                xyxy_right: Float32[ndarray, "1 4"] = right_bbox[np.newaxis, :]
                wilor_right: FinalWilorPred = hand_kpt_detector(
                    rgb_hw3=rgb_hw3,
                    xyxy=xyxy_right,
                    handedness="right",
                )
                right_uv_pred: Float32[ndarray, "1 21 2"] = wilor_right.pred_keypoints_2d.astype(np.float32, copy=False)
                uv[right_hand_idx, :] = right_uv_pred[0]
                right_conf_pred: Float32[ndarray, "1 21"] = wilor_right.confidence_2d.astype(np.float32, copy=False)
                confidences_view[right_hand_idx] = right_conf_pred[0]

            pinhole_log_path = parent_log_path / "exo" / exo_cam.name / "pinhole"
            # rr.log(
            #     f"{pinhole_log_path}/image", rr.Image(bgr_list[view_idx], color_model=rr.ColorModel.BGR).compress(70)
            # )
            confidence_rgb_view_stack: UInt8[ndarray, "1 num_kpts 3"] = confidence_scores_to_rgb(
                confidences_view[np.newaxis, :, np.newaxis]
            )
            confidence_rgb_view: UInt8[ndarray, "num_kpts 3"] = confidence_rgb_view_stack[0]
            rr.log(
                f"{pinhole_log_path}/video/keypoints",
                Points2DWithConfidence(
                    positions=uv,
                    confidences=confidences_view,
                    class_ids=0,
                    keypoint_ids=COCO_133_IDS,
                    show_labels=False,
                    colors=confidence_rgb_view,
                ),
                recording=None,
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
    calib_ts_nano: int | None = None
    """Optional nanosecond timestamp used to select calibration frames for cameras and MANO."""
    max_frames: int | None = None
    """Maximum number of frames to process. If None, all frames are processed."""


def main(config: RRDPipelineConfig) -> None:
    parent_log_path = Path("world")
    timeline = "video_time"

    ###################
    # 0. Parse inputs #
    ###################

    exoego_sequence: BaseExoEgoSequence = config.dataset.setup()  # one-liner
    rr.log("/", exoego_sequence.world_coordinate_system, static=True)
    set_annotation_context()

    parent_log_path = Path("world")
    timeline: str = "video_time"

    scene_setup_result: SceneSetupResult = setup_scene(exoego_sequence, parent_log_path, timeline)
    log_paths: LogPaths = scene_setup_result.log_paths
    shortest_timestamp: Int[ndarray, "n_frames"] = scene_setup_result.shortest_timestamp

    blueprint: rrb.Blueprint = create_blueprint(
        exo_view_roots=log_paths.exo_view_roots,
        ego_view_roots=log_paths.ego_view_roots,  # only show rgb ego cameras for now
    )
    # # show images instead of videos for now so /world/ego/camera_x/pinhole/image
    # img_exo_paths: list[Path] = [Path(p.parent.parent) / "pinhole" / "image" for p in log_paths.exo_view_roots]
    # img_ego_paths: list[Path] = [Path(p.parent.parent) / "pinhole" / "image" for p in log_paths.ego_view_roots]
    # blueprint: rrb.Blueprint = create_blueprint(
    #     exo_view_roots=img_exo_paths,
    #     ego_view_roots=img_ego_paths,
    # )
    rr.send_blueprint(blueprint)

    exo_mv_reader: MultiVideoReader = exoego_sequence.exo_sequence.exo_video_readers
    ego_mv_reader: MultiVideoReader = exoego_sequence.ego_sequence.ego_video_readers
    bgr_list_exo: list[UInt8[ndarray, "H W 3"]] = exo_mv_reader[0]
    bgr_list_ego: list[UInt8[ndarray, "H W 3"]] = ego_mv_reader[0]
    rgb_list_exo: list[UInt8[ndarray, "H W 3"]] = [cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB) for bgr in bgr_list_exo]
    rgb_list_ego: list[UInt8[ndarray, "H W 3"]] = [cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB) for bgr in bgr_list_ego]
    rgb_list: list[UInt8[ndarray, "H W 3"]] = rgb_list_exo + rgb_list_ego

    input_log_paths: list[Path] = log_paths.exo_view_roots + log_paths.ego_view_roots
    exo_ts: Int[ndarray, "num_frames"] = shortest_timestamp

    start: float = timer()

    # # Create blueprint for visualization in rerun
    # blueprint: rrb.Blueprint = create_blueprint(
    #     parent_log_path=parent_log_path, num_images=len(rgb_list), show_videos=config.videos_dir is not None
    # )
    # rr.send_blueprint(blueprint=blueprint)
    # rr.log(f"{parent_log_path}", rr.ViewCoordinates.RFU, static=True)
    rr.set_time(timeline=timeline, duration=np.timedelta64(0, "ns"))
    hand_kpt_detector: WilorHandKeypointDetector = WilorHandKeypointDetector(
        cfg=HandKeypointDetectorConfig(verbose=False)
    )

    ############################
    # 2. Calibrate Exo Cameras #
    ############################
    # for log_path, rgb in zip(log_paths.exo_view_roots, rgb_list_exo, strict=True):
    #     rr.log(
    #         f"{log_path}/pinhole/image",
    #         rr.Image(rgb, color_model=rr.ColorModel.RGB),
    #         static=True,
    #     )

    # for idx, rgb in enumerate(rgb_list_ego):
    #     rr.log(
    #         f"{parent_log_path}/ego/camera_{idx}/pinhole/image",
    #         rr.Image(rgb, color_model=rr.ColorModel.RGB),
    #         static=True,
    #     )
    mv_calibrator: MultiViewCalibrator = MultiViewCalibrator(parent_log_path=parent_log_path, config=config.calib_confg)
    mv_calib_results: MVCalibResults = mv_calibrator(rgb_list=rgb_list)

    pinhole_param_list: list[PinholeParameters] = mv_calib_results.pinhole_param_list
    exo_pinhole_param_list: list[PinholeParameters] = pinhole_param_list[: len(rgb_list_exo)]
    ego_pinhole_param_list: list[PinholeParameters] = pinhole_param_list[len(rgb_list_exo) :]
    # replace cam names with those from the dataset for easier identification
    for cam, log_path in zip(exo_pinhole_param_list, log_paths.exo_view_roots, strict=True):
        print(cam.name)
        cam.name = log_path.parent.parent.name
        print(cam.name)
    for cam, log_path in zip(ego_pinhole_param_list, log_paths.ego_view_roots, strict=True):
        print(cam.name)
        cam.name = log_path.parent.parent.name
        print(cam.name)

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
        f"{parent_log_path}/pointcloud",
        rr.Points3D(
            filtered_points,
            colors=filtered_colors,
        ),
        static=True,
    )
    #####################################
    # 4. Fuse Depths into TSDF Mesh     #
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
            str(parent_log_path / "gt_mesh"),
            rr.Mesh3D(
                vertex_positions=vertex_positions,
                triangle_indices=triangle_indices,
                vertex_normals=vertex_normals,
                vertex_colors=vertex_colors,
            ),
            static=True,
        )

    # 6. Predict keypoints from calibrated video frames
    if exo_mv_reader is not None and exo_ts is not None:
        xyzc_list: list[Float32[ndarray, "num_kpts 4"]] = predict_kpts3d_from_calibrated_videos(
            exo_video_readers=exo_mv_reader,
            exo_cam_list=exo_pinhole_param_list,
            shortest_timestamp=exo_ts,
            parent_log_path=parent_log_path,
            hand_kpt_detector=hand_kpt_detector,
            max_frames=config.max_frames,
        )
        upper_body_filter_idx = np.array([5, 6, 7, 8, 9, 10])
        face_idx = np.arange(23, 91)
        left_hand_idx = np.arange(91, 112)
        right_hand_idx = np.arange(112, 133)
        wb_upper_body_filter_idx = np.concatenate([upper_body_filter_idx, face_idx, left_hand_idx, right_hand_idx])
        top_half_mask = np.isin(np.arange(133), wb_upper_body_filter_idx)
        bbox_expansion_percentage: float = 0.25
        # project into ego views
        if len(xyzc_list) > 0:
            exo_fps: float = float(exo_mv_reader.video_readers[0].fps) if exo_mv_reader.video_readers else 0.0
            ego_fps: float = float(ego_mv_reader.video_readers[0].fps) if ego_mv_reader.video_readers else 0.0
            ego_frame_count: int = len(ego_mv_reader)
            frame_ratio: float = (
                ego_fps / exo_fps
                if exo_fps > 0.0 and ego_fps > 0.0
                else (ego_frame_count / float(len(xyzc_list)) if len(xyzc_list) > 0 else 1.0)
            )
            frame_ratio = 1.0 if not np.isfinite(frame_ratio) or frame_ratio <= 0.0 else frame_ratio
            Pall_ego: Float32[ndarray, "n_views 3 4"] = np.stack(
                [cam.projection_matrix for cam in ego_pinhole_param_list]
            ).astype(np.float32)
            for idx, xyzc in enumerate(xyzc_list):
                rr.set_time(timeline=timeline, duration=np.timedelta64(int(exo_ts[idx]), "ns"))
                # if ego_frame_count == 0:
                #     continue
                # ego_idx_float: float = idx * frame_ratio
                # ego_frame_idx: int = min(int(round(ego_idx_float)), ego_frame_count - 1)
                bgr_list_ego: list[UInt8[ndarray, "H W 3"]] = ego_mv_reader[idx]

                if xyzc is None:
                    continue
                vis_xyz: Float32[ndarray, "num_kpts 3"] = xyzc[:, :3].copy()
                vis_scores_3d: Float32[ndarray, "num_kpts"] = xyzc[:, 3].copy()  # noqa: UP037
                # filter to only include the desired keypoints
                vis_xyz[~top_half_mask, :] = np.nan
                vis_scores_3d[~top_half_mask] = np.nan
                # filter out low-confidence keypoints
                vis_xyz[vis_scores_3d < 0.7, :] = np.nan
                vis_scores_3d[vis_scores_3d < 0.7] = np.nan

                # project 3d keypoints into 2d and log
                xyz_hom: Float32[ndarray, "num_kpts 4"] = np.concatenate(
                    [vis_xyz, np.ones((vis_xyz.shape[0], 1), dtype=np.float32)], axis=1
                )
                xyz_hom_stack: Float32[ndarray, "1 num_kpts 4"] = np.stack([xyz_hom], axis=0)
                uv_ego_stack: Float[ndarray, "1 n_views 133 2"] = proj_3d_vectorized(xyz_hom=xyz_hom_stack, P=Pall_ego)
                uv_ego: Float32[ndarray, "n_views 133 2"] = uv_ego_stack[0]

                for view_idx, (uv_view, ego_cam) in enumerate(zip(uv_ego, ego_pinhole_param_list, strict=True)):
                    pinhole_log_path: Path = parent_log_path / "ego" / ego_cam.name / "pinhole"
                    uv: Float32[ndarray, "133 2"] = uv_view.astype(np.float32, copy=True)
                    # filter out keypoints that are behind the camera
                    uv[vis_xyz[:, 2] > 0, :] = np.nan
                    # filter out keypoints that are out of bounds
                    uv = filter_out_of_bounds_keypoints(uv, ego_cam, margin_percentage=0.0)

                    confidences_view: Float32[ndarray, "133"] = vis_scores_3d.astype(np.float32, copy=True)
                    rgb_hw3: UInt8[ndarray, "H W 3"] = bgr_list_ego[view_idx][..., ::-1]

                    left_bbox_infer: Float32[ndarray, "4"] | None = compute_square_bbox(
                        uv[left_hand_idx, :],
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
                        uv[left_hand_idx, :] = left_uv_pred[0]
                        left_conf_pred: Float32[ndarray, "1 21"] = wilor_left.confidence_2d.astype(
                            np.float32, copy=False
                        )
                        confidences_view[left_hand_idx] = left_conf_pred[0]

                    right_bbox_infer: Float32[ndarray, "4"] | None = compute_square_bbox(
                        uv[right_hand_idx, :],
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
                        uv[right_hand_idx, :] = right_uv_pred[0]
                        right_conf_pred: Float32[ndarray, "1 21"] = wilor_right.confidence_2d.astype(
                            np.float32, copy=False
                        )
                        confidences_view[right_hand_idx] = right_conf_pred[0]

                    uv = filter_out_of_bounds_keypoints(uv, ego_cam, margin_percentage=0.0)

                    left_bbox_log: Float32[ndarray, "4"] | None = compute_square_bbox(
                        uv[left_hand_idx, :],
                        intrinsics=ego_cam.intrinsics,
                        expansion_ratio=bbox_expansion_percentage,
                    )
                    if left_bbox_log is not None:
                        lh_xyxy: Float32[ndarray, "4"] = left_bbox_log.astype(np.float32, copy=False)
                        rr.log(
                            f"{pinhole_log_path}/video/left_hand_bbox",
                            rr.Boxes2D(array=lh_xyxy, array_format=rr.Box2DFormat.XYXY),
                            recording=None,
                        )
                    else:
                        rr.log(f"{pinhole_log_path}/video/left_hand_bbox", rr.Clear(recursive=True))

                    right_bbox_log: Float32[ndarray, "4"] | None = compute_square_bbox(
                        uv[right_hand_idx, :],
                        intrinsics=ego_cam.intrinsics,
                        expansion_ratio=bbox_expansion_percentage,
                    )
                    if right_bbox_log is not None:
                        rh_xyxy: Float32[ndarray, "4"] = right_bbox_log.astype(np.float32, copy=False)
                        rr.log(
                            f"{pinhole_log_path}/video/right_hand_bbox",
                            rr.Boxes2D(array=rh_xyxy, array_format=rr.Box2DFormat.XYXY),
                            recording=None,
                        )
                    else:
                        rr.log(f"{pinhole_log_path}/video/right_hand_bbox", rr.Clear(recursive=True))

                    # rr.log(
                    #     f"{pinhole_log_path}/image",
                    #     rr.Image(bgr_list_ego[view_idx], color_model=rr.ColorModel.BGR).compress(70),
                    # )

                    confidence_rgb_view_stack: UInt8[ndarray, "1 num_kpts 3"] = confidence_scores_to_rgb(
                        confidences_view[np.newaxis, :, np.newaxis]
                    )
                    confidence_rgb_view: UInt8[ndarray, "num_kpts 3"] = confidence_rgb_view_stack[0]
                    rr.log(
                        f"{pinhole_log_path}/video/keypoints",
                        Points2DWithConfidence(
                            positions=uv,
                            confidences=confidences_view,
                            class_ids=0,
                            keypoint_ids=COCO_133_IDS,
                            show_labels=False,
                            colors=confidence_rgb_view,
                        ),
                        recording=None,
                    )

    print(f"Inference completed in {timer() - start:.2f} seconds")
