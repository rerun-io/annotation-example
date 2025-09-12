from dataclasses import dataclass, replace
from pathlib import Path
from timeit import default_timer as timer
from typing import Literal

import cv2
import numpy as np
import open3d as o3d
import rerun as rr
import rerun.blueprint as rrb
import torch
from einops import rearrange
from jaxtyping import Bool, Float, Float32, Int, UInt8
from monopriors.depth_utils import depth_edges_mask, multidepth_to_points
from monopriors.multiview_models.vggt_model import MultiviewPred, VGGTPredictor, robust_filter_confidences
from monopriors.relative_depth_models import (
    RelativeDepthPrediction,
    get_relative_predictor,
)
from monopriors.relative_depth_models.base_relative_depth import BaseRelativePredictor
from monopriors.scale_utils import compute_scale_and_shift
from numpy import ndarray
from simplecv.camera_orient_utils import auto_orient_and_center_poses
from simplecv.camera_parameters import Extrinsics
from simplecv.ops.conventions import CameraConventions, convert_pose
from simplecv.rerun_log_utils import RerunTyroConfig, log_pinhole, log_video
from simplecv.video_io import MultiVideoReader
from tqdm.auto import trange

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
                origin=f"{parent_log_path}/camera_{camera_index}/pinhole/depth",
                contents=[
                    "+ $origin/**",
                ],
                name="Depth",
            ),
            rrb.Spatial2DView(
                origin=f"{parent_log_path}/camera_{camera_index}/pinhole/filtered_depth",
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
                origin=f"{parent_log_path}/camera_{camera_index}/pinhole/image",
                contents=[
                    "+ $origin/**",
                ],
                name="Image Content",
            ),
            create_depth_views(parent_log_path, camera_index),
            rrb.Spatial2DView(
                origin=f"{parent_log_path}/camera_{camera_index}/pinhole/confidence",
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


def create_blueprint(parent_log_path: Path, num_images: int, show_videos: bool = False) -> rrb.Blueprint:
    view3d = rrb.Spatial3DView(
        origin=f"{parent_log_path}",
        contents=[
            "+ $origin/**",
            # don't include depths in the 3D view, as they can be very noisy
            *[f"- /{parent_log_path}/camera_{i}/pinhole/depth" for i in range(num_images)],
            *[f"- /{parent_log_path}/camera_{i}/pinhole/filtered_depth" for i in range(num_images)],
            *[f"- /{parent_log_path}/camera_{i}/pinhole/refined_depth" for i in range(num_images)],
            *[f"- /{parent_log_path}/camera_{i}/pinhole/confidence" for i in range(num_images)],
            *[f"- /{parent_log_path}/camera_{i}/pinhole/image" for i in range(num_images)],
        ],
        line_grid=rrb.archetypes.LineGrid3D(visible=False),
    )

    # Create tabbed view that supports any number of cameras
    view_2d: rrb.Tabs = create_tabbed_camera_view(parent_log_path, num_images)
    if show_videos:
        view_2d_videos: rrb.Grid = rrb.Grid(
            contents=[
                rrb.Spatial2DView(origin=f"{parent_log_path}/camera_{i}/pinhole/video", name=f"Video {i + 1}")
                for i in range(num_images)
            ],
            name="Videos Tab",
        )
        view_2d = rrb.Tabs(view_2d, view_2d_videos)

    blueprint = rrb.Blueprint(rrb.Horizontal(contents=[view3d, view_2d], column_shares=[3, 2]), collapse_panels=True)
    return blueprint


def orient_mv_pred_list(
    mv_pred_list: list[MultiviewPred],
    method: Literal["pca", "up", "vertical", "none"] = "up",
    center_method: Literal["poses", "focus", "none"] = "none",
) -> list[MultiviewPred]:
    extri_list: list[Extrinsics] = [mv_pred.pinhole_param.extrinsics for mv_pred in mv_pred_list]

    world_T_cam_batch: Float[ndarray, "*num_poses 4 4"] = np.stack([extri.world_T_cam for extri in extri_list])
    assert len(set(mv_pred.pinhole_param.intrinsics.camera_conventions for mv_pred in mv_pred_list)) == 1
    if mv_pred_list[0].pinhole_param.intrinsics.camera_conventions == "RDF":
        world_T_cam_gl: Float[ndarray, "*num_poses 4 4"] = convert_pose(
            world_T_cam_batch, CameraConventions.CV, CameraConventions.GL
        )
    else:
        world_T_cam_gl = world_T_cam_batch

    # NumPy-only orientation (returns (N,3,4))
    oriented_world_T_cam_3x4_np, _ = auto_orient_and_center_poses(
        world_T_cam_gl.astype(np.float64), method=method, center_method=center_method
    )

    N: int = oriented_world_T_cam_3x4_np.shape[0]
    bottom_row: Float[ndarray, "N 1 4"] = np.broadcast_to(np.array([[0.0, 0.0, 0.0, 1.0]]), (N, 1, 4))
    oriented_world_T_cam_4x4_np: Float32[ndarray, "N 4 4"] = np.concatenate(
        [oriented_world_T_cam_3x4_np, bottom_row], axis=1
    ).astype(np.float32)
    oriented_world_T_cam_cv: Float[ndarray, "N 4 4"] = convert_pose(
        oriented_world_T_cam_4x4_np, CameraConventions.GL, CameraConventions.CV
    )
    # put back into mv pred list using replace
    oriented_mv_pred_list: list[MultiviewPred] = []
    for idx, mv_pred in enumerate(mv_pred_list):
        oriented_extri: Extrinsics = Extrinsics(
            world_R_cam=oriented_world_T_cam_cv[idx, :3, :3],
            world_t_cam=oriented_world_T_cam_cv[idx, :3, 3],
        )
        oriented_mv_pred_list.append(
            replace(mv_pred, pinhole_param=replace(mv_pred.pinhole_param, extrinsics=oriented_extri))
        )

    return oriented_mv_pred_list


def estimate_voxel_size(
    points: Float32[ndarray, "N 3"],
    target_points: int = 100_000,
    tolerance: float = 0.25,
    max_iterations: int = 10,
    min_voxel_ratio: float = 0.0001,
    max_voxel_ratio: float = 0.5,
) -> float:
    """
    Use binary search to find optimal voxel size for target point count.

    Args:
        points: Input point cloud points
        target_points: Desired number of points after downsampling
        tolerance: Acceptable relative error (0.25 = within 25% of target)
        max_iterations: Maximum binary search iterations
        min_voxel_ratio: Minimum voxel size as ratio of scene diagonal
        max_voxel_ratio: Maximum voxel size as ratio of scene diagonal

    Returns:
        Voxel size that results in point count within tolerance of target_points
    """
    if len(points) == 0:
        return 0.01  # Default fallback

    # Calculate scene bounds for voxel size limits
    min_bounds: Float32[ndarray, "3"] = np.min(points, axis=0)
    max_bounds: Float32[ndarray, "3"] = np.max(points, axis=0)
    scene_diagonal: float = float(np.linalg.norm(max_bounds - min_bounds))

    # Set search bounds
    min_voxel_size: float = scene_diagonal * min_voxel_ratio
    max_voxel_size: float = scene_diagonal * max_voxel_ratio

    # Create Open3D point cloud once for reuse
    pcd_temp: o3d.geometry.PointCloud = o3d.geometry.PointCloud()
    pcd_temp.points = o3d.utility.Vector3dVector(points)

    # Binary search for optimal voxel size
    low: float = min_voxel_size
    high: float = max_voxel_size
    best_voxel_size: float = (low + high) / 2

    t = trange(max_iterations, desc="Estimating voxel size")
    for _ in t:
        current_voxel_size: float = (low + high) / 2

        # Test this voxel size
        pcd_test: o3d.geometry.PointCloud = pcd_temp.voxel_down_sample(current_voxel_size)
        current_points: int = len(pcd_test.points)

        # Calculate relative error
        error: float = abs(current_points - target_points) / target_points

        # update progress bar postfix
        t.set_postfix(
            {
                "voxel_size": f"{current_voxel_size:.6f}",
                "points": current_points,
                "error": f"{error:.3f}",
            }
        )

        # Check if we're within tolerance
        if error <= tolerance:
            best_voxel_size = current_voxel_size
            t.write(f"  - ✓ Found optimal voxel size: {best_voxel_size:.6f}")
            break

        # Update search bounds
        if current_points > target_points:
            # Too many points, need larger voxel size
            low = current_voxel_size
        else:
            # Too few points, need smaller voxel size
            high = current_voxel_size

        best_voxel_size = current_voxel_size

    return float(best_voxel_size)


def mv_pred_to_pointcloud(
    mv_pred_list: list[MultiviewPred], depth_list: list[Float32[ndarray, "H W"]] | None = None
) -> Float32[ndarray, "num_points 3"]:
    if depth_list is None:
        depth_maps: Float32[ndarray, "b h w 1"] = np.stack(
            [rearrange(mv_pred.depth_map, "h w -> h w 1") for mv_pred in mv_pred_list], axis=0
        ).astype(np.float32)
    else:
        depth_maps: Float32[ndarray, "b h w 1"] = np.stack(
            [rearrange(depth, "h w -> h w 1") for depth in depth_list], axis=0
        ).astype(np.float32)

    # multidepth_to_points requires world_T_cam not cam_T_world
    world_T_cam_b44: Float32[ndarray, "num_cams 4 4"] = np.stack(
        [mv_pred.pinhole_param.extrinsics.world_T_cam for mv_pred in mv_pred_list], axis=0
    ).astype(np.float32)
    K_b33: Float32[ndarray, "b 3 3"] = np.stack(
        [mv_pred.pinhole_param.intrinsics.k_matrix for mv_pred in mv_pred_list], axis=0
    ).astype(np.float32)
    world_points: Float32[ndarray, "b h w 3"] = multidepth_to_points(
        depth_maps=depth_maps, world_T_cam_batch=world_T_cam_b44, K_b33=K_b33
    )
    pointcloud: Float32[ndarray, "num_points 3"] = world_points.reshape(-1, 3)
    return pointcloud


@dataclass
class VGGTInferenceConfig:
    rr_config: RerunTyroConfig
    image_dir: Path | None = None
    """Directory containing input images."""
    videos_dir: Path | None = None
    """Directory containing input videos."""
    ts_idx: int = 0
    """Timestep for video chosen frames."""
    keep_top_percent: int | float = 30.0
    """keep_top_percent: Percentage in [0,100]. Interpreted as the fraction to discard;
        the top (100 - keep_top_percent)% of pixel scores are kept.
        E.g. 75 -> keep top 25%; 30 -> keep top 70%."""
    preprocessing_mode: Literal["crop", "pad"] = "crop"
    """Mode for image preprocessing: 'crop' preserves aspect ratio, 'pad' adds white padding"""
    refine_depth_maps: bool = False
    """Whether to refine depth maps during processing. To make them metric"""
    max_frames: int | None = None
    """Maximum number of frames to process. If None, all frames are processed."""
    output_dir: Path | None = None
    """Output directory for colmap version. If None, results are not saved."""


def main(config: VGGTInferenceConfig) -> None:
    parent_log_path = Path("world")
    timeline = "video_time"

    if config.image_dir is None and config.videos_dir is None:
        raise ValueError("Either image or videos directory must be specified")

    ###################
    # 0. Parse inputs #
    ###################
    input_type: Literal["videos", "images"] = "images" if config.image_dir is not None else "videos"
    match input_type:
        case "images":
            image_paths = []

            for ext in SUPPORTED_IMAGE_EXTENSIONS:
                image_paths.extend(config.image_dir.glob(f"*{ext}"))
            image_paths: list[Path] = sorted(image_paths)
            assert len(image_paths) > 0, (
                f"No images found in {config.image_dir} in supported formats {SUPPORTED_IMAGE_EXTENSIONS}"
            )

            bgr_list: list[UInt8[ndarray, "H W 3"]] = [cv2.imread(str(image_path)) for image_path in image_paths]
        case "videos":
            video_path_list: list[Path] = sorted(config.videos_dir.glob("*.mp4"))
            assert len(video_path_list) > 0, f"No videos found in {config.videos_dir}"
            exo_timestamps: list[Int[ndarray, "num_frames"]] = []
            for i, video_path in enumerate(video_path_list):
                frame_timestamps_ns: Int[ndarray, "num_frames"] = log_video(
                    video_path=video_path,
                    video_log_path=parent_log_path / f"camera_{i}" / "pinhole" / "video",
                    timeline=timeline,
                )
                exo_timestamps.append(frame_timestamps_ns)

            mv_reader = MultiVideoReader(video_path_list)
            bgr_list: list[UInt8[ndarray, "H W 3"]] = mv_reader[config.ts_idx]

    rgb_list: list[UInt8[ndarray, "H W 3"]] = [cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB) for bgr in bgr_list]

    start: float = timer()

    blueprint: rrb.Blueprint = create_blueprint(
        parent_log_path=parent_log_path, num_images=len(rgb_list), show_videos=config.videos_dir is not None
    )
    rr.send_blueprint(blueprint=blueprint)
    rr.log(f"{parent_log_path}", rr.ViewCoordinates.RFU, static=True)
    rr.set_time(timeline, duration=0)

    vggt_predictor = VGGTPredictor(
        device=device,
        preprocessing_mode=config.preprocessing_mode,
    )
    mv_pred_list: list[MultiviewPred] = vggt_predictor(rgb_list=rgb_list)
    mv_pred_list = orient_mv_pred_list(mv_pred_list)

    pointcloud: Float32[ndarray, "num_points 3"] = mv_pred_to_pointcloud(mv_pred_list)
    rgb_stack: UInt8[ndarray, "num_points 3"] = np.concatenate(
        [rearrange(mv_pred.rgb_image, "h w c -> (h w) c") for mv_pred in mv_pred_list]
    )

    # create depth confidence values using robust filtering for top keep percentile
    depth_confidences: list[UInt8[ndarray, "H W"]] = [
        robust_filter_confidences(mv_pred.confidence_mask, keep_top_percent=config.keep_top_percent)
        for mv_pred in mv_pred_list
    ]

    # new_depth_confidences = depth_confidences
    pc_conf_mask: Bool[ndarray, "num_points"] = np.concatenate(
        [rearrange(depth_conf, "h w -> (h w)") for depth_conf in depth_confidences]
    ).astype(bool)

    # Filter by confidence BEFORE downsampling for better quality and efficiency
    filtered_points_pre_ds: Float32[ndarray, "filtered_points 3"] = pointcloud[pc_conf_mask]
    filtered_colors_pre_ds: UInt8[ndarray, "filtered_points 3"] = rgb_stack[pc_conf_mask]

    # Create point cloud from high-confidence points only
    pcd: o3d.geometry.PointCloud = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(filtered_points_pre_ds)
    pcd.colors = o3d.utility.Vector3dVector(filtered_colors_pre_ds / 255.0)  # Open3D expects [0,1] range

    # Automatically determine optimal voxel size based on point cloud characteristics
    voxel_size: float = estimate_voxel_size(filtered_points_pre_ds, target_points=200_000)
    pcd_ds: o3d.geometry.PointCloud = pcd.voxel_down_sample(voxel_size)

    filtered_points: Float32[ndarray, "final_points 3"] = np.asarray(pcd_ds.points, dtype=np.float32)
    filtered_colors: Float32[ndarray, "final_points 3"] = np.asarray(pcd_ds.colors, dtype=np.float32)

    rr.log(
        f"{parent_log_path}/point_cloud",
        rr.Points3D(
            filtered_points,
            colors=filtered_colors,
        ),
        static=True,
    )
    if config.refine_depth_maps:
        predictor: BaseRelativePredictor = get_relative_predictor("MogeV1Predictor")(device="cuda")
        refined_depths_list: list[Float32[ndarray, "H W"]] = []
    mv_pred: MultiviewPred
    for mv_pred in mv_pred_list:
        cam_log_path: Path = parent_log_path / mv_pred.cam_name
        pinhole_log_path: Path = cam_log_path / "pinhole"

        depth_map: Float32[ndarray, "H W"] = mv_pred.depth_map
        depth_conf: UInt8[ndarray, "H W"] = depth_confidences[mv_pred_list.index(mv_pred)]
        # Filter depth
        filtered_depth_map: Float32[ndarray, "H W"] = np.where(depth_conf > 0, depth_map, 0)

        if config.refine_depth_maps:
            relative_pred: RelativeDepthPrediction = predictor.__call__(
                rgb=mv_pred.rgb_image, K_33=mv_pred.pinhole_param.intrinsics.k_matrix
            )

            scale, shift = compute_scale_and_shift(
                relative_pred.depth, filtered_depth_map, mask=depth_conf > 0, scale_only=False
            )
            metric_depth: Float32[np.ndarray, "h w"] = relative_pred.depth.copy() * scale + shift
            # filter depth
            edges_mask: Bool[np.ndarray, "h w"] = depth_edges_mask(metric_depth, threshold=0.01)
            metric_depth: Float32[np.ndarray, "h w"] = metric_depth * ~edges_mask

            refined_depths_list.append(metric_depth)

        log_pinhole(
            mv_pred.pinhole_param,
            cam_log_path=cam_log_path,
            image_plane_distance=0.05,
            static=True,
        )

        rr.log(
            f"{pinhole_log_path}/image",
            rr.Image(mv_pred.rgb_image, color_model=rr.ColorModel.RGB).compress(),
            static=True,
        )
        rr.log(
            f"{pinhole_log_path}/confidence",
            rr.Image(depth_conf, color_model=rr.ColorModel.L).compress(),
            static=True,
        )
        rr.log(
            f"{pinhole_log_path}/filtered_depth",
            rr.DepthImage(filtered_depth_map, meter=1),
            static=True,
        )
        rr.log(
            f"{pinhole_log_path}/depth",
            rr.DepthImage(depth_map, meter=1),
            static=True,
        )
        if config.refine_depth_maps:
            rr.log(
                f"{pinhole_log_path}/refined_depth",
                rr.DepthImage(metric_depth, meter=1),
                static=True,
            )

    if config.refine_depth_maps:
        moge_points: Float32[ndarray, "num_points 3"] = mv_pred_to_pointcloud(
            mv_pred_list, depth_list=refined_depths_list
        )
        new_pc: Float32[ndarray, "num_points 3"] = moge_points.reshape(-1, 3)
        rgb_stack: UInt8[ndarray, "num_points 3"] = np.concatenate(
            [rearrange(mv_pred.rgb_image, "h w c -> (h w) c") for mv_pred in mv_pred_list]
        )

        # Create point cloud from high-confidence points only
        pcd: o3d.geometry.PointCloud = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(new_pc)
        pcd.colors = o3d.utility.Vector3dVector(rgb_stack / 255.0)  # Open3D expects [0,1] range

        # Automatically determine optimal voxel size based on point cloud characteristics
        voxel_size: float = estimate_voxel_size(new_pc, target_points=500_000)
        pcd_ds = pcd.voxel_down_sample(voxel_size)

        filtered_points: Float32[ndarray, "final_points 3"] = np.asarray(pcd_ds.points, dtype=np.float32)
        filtered_colors: Float32[ndarray, "final_points 3"] = np.asarray(pcd_ds.colors, dtype=np.float32)

        rr.log(
            f"{parent_log_path}/moge_point_cloud",
            rr.Points3D(
                filtered_points,
                colors=filtered_colors,
            ),
            static=True,
        )

    print(f"Inference completed in {timer() - start:.2f} seconds")


def unload_vggt_model(vggt_predictor: VGGTPredictor) -> None:
    """
    Unload the VGGT model from memory by moving it to CPU and deleting references.
    This helps free GPU memory more effectively.

    Args:
        vggt_predictor: The VGGTPredictor instance containing the model to unload
    """
    # Move model to CPU first (helps with GPU memory cleanup)
    if hasattr(vggt_predictor, "model") and vggt_predictor.model is not None:
        vggt_predictor.model.cpu()
        # Remove reference to model
        vggt_predictor.model = None
        # Clear GPU cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print("VGGT model unloaded from memory")
