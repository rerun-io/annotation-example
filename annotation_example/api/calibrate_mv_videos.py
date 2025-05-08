import math
import warnings
from dataclasses import dataclass
from pathlib import Path
from timeit import default_timer as timer
from typing import Literal

import cv2
import numpy as np
import open3d as o3d
import rerun as rr
import rerun.blueprint as rrb
import torch
from einops import repeat
from jaxtyping import Float, Float32, UInt8, UInt16
from monopriors.multiview_models.vggt_model import MultiviewPred, VGGTPredictor
from numpy import ndarray
from simplecv.camera_parameters import Extrinsics
from simplecv.conversion_utils import save_to_nerfstudio
from simplecv.rerun_log_utils import RerunTyroConfig, log_pinhole
from simplecv.video_io import MultiVideoReader
from torch import Tensor

# turn off the torch.cuda.amp.autocast FutureWarning
warnings.filterwarnings("ignore", category=FutureWarning, message=r".*torch\.cuda\.amp\.autocast.*")


def create_blueprint(parent_log_path: Path, image_paths: list[Path]) -> rrb.Blueprint:
    view3d = rrb.Spatial3DView(
        origin=f"{parent_log_path}",
    )

    blueprint = rrb.Blueprint(rrb.Horizontal(contents=[view3d]), collapse_panels=True)
    return blueprint


@dataclass
class CalibrateConfig:
    rr_config: RerunTyroConfig
    video_dir: Path = Path("data/synchronized/multicam")
    device: Literal["cpu", "cuda"] = "cuda"
    confidence_threshold: float = 75.0
    output_dir: Path | None = None


def rotation_matrix_between(a: Float[Tensor, "3"], b: Float[Tensor, "3"]) -> Float[Tensor, "3 3"]:
    """Compute the rotation matrix that rotates vector a to vector b.

    Args:
        a: The vector to rotate.
        b: The vector to rotate to.
    Returns:
        The rotation matrix.
    """
    a = a / torch.linalg.norm(a)
    b = b / torch.linalg.norm(b)
    v = torch.linalg.cross(a, b)  # Axis of rotation.

    # Handle cases where `a` and `b` are parallel.
    eps = 1e-6
    if torch.sum(torch.abs(v)) < eps:
        x = torch.tensor([1.0, 0, 0]) if abs(a[0]) < eps else torch.tensor([0, 1.0, 0])
        v = torch.linalg.cross(a, x)

    v = v / torch.linalg.norm(v)
    skew_sym_mat = torch.Tensor(
        [
            [0, -v[2], v[1]],
            [v[2], 0, -v[0]],
            [-v[1], v[0], 0],
        ]
    )
    theta = torch.acos(torch.clip(torch.dot(a, b), -1, 1))

    # Rodrigues rotation formula. https://en.wikipedia.org/wiki/Rodrigues%27_rotation_formula
    return torch.eye(3) + torch.sin(theta) * skew_sym_mat + (1 - torch.cos(theta)) * (skew_sym_mat @ skew_sym_mat)


def focus_of_attention(poses: Float[Tensor, "*num_poses 4 4"], initial_focus: Float[Tensor, "3"]) -> Float[Tensor, "3"]:
    """Compute the focus of attention of a set of cameras. Only cameras
    that have the focus of attention in front of them are considered.

     Args:
        poses: The poses to orient.
        initial_focus: The 3D point views to decide which cameras are initially activated.

    Returns:
        The 3D position of the focus of attention.
    """
    # References to the same method in third-party code:
    # https://github.com/google-research/multinerf/blob/1c8b1c552133cdb2de1c1f3c871b2813f6662265/internal/camera_utils.py#L145
    # https://github.com/bmild/nerf/blob/18b8aebda6700ed659cb27a0c348b737a5f6ab60/load_llff.py#L197
    active_directions = -poses[:, :3, 2:3]
    active_origins = poses[:, :3, 3:4]
    # initial value for testing if the focus_pt is in front or behind
    focus_pt = initial_focus
    # Prune cameras which have the current have the focus_pt behind them.
    active = torch.sum(active_directions.squeeze(-1) * (focus_pt - active_origins.squeeze(-1)), dim=-1) > 0
    done = False
    # We need at least two active cameras, else fallback on the previous solution.
    # This may be the "poses" solution if no cameras are active on first iteration, e.g.
    # they are in an outward-looking configuration.
    while torch.sum(active.int()) > 1 and not done:
        active_directions = active_directions[active]
        active_origins = active_origins[active]
        # https://en.wikipedia.org/wiki/Line–line_intersection#In_more_than_two_dimensions
        m = torch.eye(3) - active_directions * torch.transpose(active_directions, -2, -1)
        mt_m = torch.transpose(m, -2, -1) @ m
        focus_pt = torch.linalg.inv(mt_m.mean(0)) @ (mt_m @ active_origins).mean(0)[:, 0]
        active = torch.sum(active_directions.squeeze(-1) * (focus_pt - active_origins.squeeze(-1)), dim=-1) > 0
        if active.all():
            # the set of active cameras did not change, so we're done.
            done = True
    return focus_pt


def auto_orient_and_center_poses(
    poses: Float[Tensor, "*num_poses 4 4"],
    method: Literal["pca", "up", "vertical", "none"] = "up",
    center_method: Literal["poses", "focus", "none"] = "poses",
) -> tuple[Float[Tensor, "*num_poses 3 4"], Float[Tensor, "3 4"]]:
    """Orients and centers the poses.

    We provide three methods for orientation:

    - pca: Orient the poses so that the principal directions of the camera centers are aligned
        with the axes, Z corresponding to the smallest principal component.
        This method works well when all of the cameras are in the same plane, for example when
        images are taken using a mobile robot.
    - up: Orient the poses so that the average up vector is aligned with the z axis.
        This method works well when images are not at arbitrary angles.
    - vertical: Orient the poses so that the Z 3D direction projects close to the
        y axis in images. This method works better if cameras are not all
        looking in the same 3D direction, which may happen in camera arrays or in LLFF.

    There are two centering methods:

    - poses: The poses are centered around the origin.
    - focus: The origin is set to the focus of attention of all cameras (the
        closest point to cameras optical axes). Recommended for inward-looking
        camera configurations.

    Args:
        poses: The poses to orient.
        method: The method to use for orientation.
        center_method: The method to use to center the poses.

    Returns:
        Tuple of the oriented poses and the transform matrix.
    """

    origins = poses[..., :3, 3]

    mean_origin = torch.mean(origins, dim=0)
    translation_diff = origins - mean_origin

    if center_method == "poses":
        translation = mean_origin
    elif center_method == "focus":
        translation = focus_of_attention(poses, mean_origin)
    elif center_method == "none":
        translation = torch.zeros_like(mean_origin)
    else:
        raise ValueError(f"Unknown value for center_method: {center_method}")

    if method == "pca":
        _, eigvec = torch.linalg.eigh(translation_diff.T @ translation_diff)
        eigvec = torch.flip(eigvec, dims=(-1,))

        if torch.linalg.det(eigvec) < 0:
            eigvec[:, 2] = -eigvec[:, 2]

        transform = torch.cat([eigvec, eigvec @ -translation[..., None]], dim=-1)
        oriented_poses = transform @ poses

        if oriented_poses.mean(dim=0)[2, 1] < 0:
            oriented_poses[1:3, :] = -1 * oriented_poses[1:3, :]
            transform[1:3, :] = -1 * transform[1:3, :]
    elif method in ("up", "vertical"):
        up = torch.mean(poses[:, :3, 1], dim=0)
        up = up / torch.linalg.norm(up)
        if method == "vertical":
            # If cameras are not all parallel (e.g. not in an LLFF configuration),
            # we can find the 3D direction that most projects vertically in all
            # cameras by minimizing ||Xu|| s.t. ||u||=1. This total least squares
            # problem is solved by SVD.
            x_axis_matrix = poses[:, :3, 0]
            _, S, Vh = torch.linalg.svd(x_axis_matrix, full_matrices=False)
            # Singular values are S_i=||Xv_i|| for each right singular vector v_i.
            # ||S|| = sqrt(n) because lines of X are all unit vectors and the v_i
            # are an orthonormal basis.
            # ||Xv_i|| = sqrt(sum(dot(x_axis_j,v_i)^2)), thus S_i/sqrt(n) is the
            # RMS of cosines between x axes and v_i. If the second smallest singular
            # value corresponds to an angle error less than 10° (cos(80°)=0.17),
            # this is probably a degenerate camera configuration (typical values
            # are around 5° average error for the true vertical). In this case,
            # rather than taking the vector corresponding to the smallest singular
            # value, we project the "up" vector on the plane spanned by the two
            # best singular vectors. We could also just fallback to the "up"
            # solution.
            if S[1] > 0.17 * math.sqrt(poses.shape[0]):
                # regular non-degenerate configuration
                up_vertical = Vh[2, :]
                # It may be pointing up or down. Use "up" to disambiguate the sign.
                up = up_vertical if torch.dot(up_vertical, up) > 0 else -up_vertical
            else:
                # Degenerate configuration: project "up" on the plane spanned by
                # the last two right singular vectors (which are orthogonal to the
                # first). v_0 is a unit vector, no need to divide by its norm when
                # projecting.
                up = up - Vh[0, :] * torch.dot(up, Vh[0, :])
                # re-normalize
                up = up / torch.linalg.norm(up)

        rotation = rotation_matrix_between(up, torch.Tensor([0, 0, 1]))
        transform = torch.cat([rotation, rotation @ -translation[..., None]], dim=-1)
        oriented_poses = transform @ poses
    elif method == "none":
        transform = torch.eye(4)
        transform[:3, 3] = -translation
        transform = transform[:3, :]
        oriented_poses = transform @ poses
    else:
        raise ValueError(f"Unknown value for method: {method}")

    return oriented_poses, transform


def calibrate_mv_videos(config: CalibrateConfig):
    parent_log_path: Path = Path("world")
    timeline_name = "frame_idx"
    video_paths: list[Path] = sorted(config.video_dir.glob("*.mp4"))
    if len(video_paths) == 0:
        raise ValueError(f"No videos found in {config.video_dir}")

    # load in multiview videos
    mv_reader = MultiVideoReader(video_paths=video_paths)

    # setup rerun
    blueprint: rrb.Blueprint = create_blueprint(parent_log_path=parent_log_path, image_paths=video_paths)
    rr.send_blueprint(blueprint)

    rr.log(f"{parent_log_path}", rr.ViewCoordinates.RIGHT_HAND_Z_DOWN, static=True)
    rr.set_time(timeline_name, sequence=0)

    # get the first frame of each video
    bgr_list: list[UInt8[ndarray, "H W 3"]] = mv_reader[0]
    rgb_list: list[UInt8[ndarray, "H W 3"]] = [cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB) for bgr in bgr_list]

    # predict the multiview poses + point clouds
    device = "cuda" if torch.cuda.is_available() else "cpu"
    vggt_predictor = VGGTPredictor(
        device=config.device,
        confidence_threshold=config.confidence_threshold,
        preprocessing_mode="crop",
    )
    inference_start: float = timer()
    multiview_predictions: list[MultiviewPred] = vggt_predictor(rgb_list=rgb_list)
    print(f"Inference time: {timer() - inference_start:.2f} seconds")

    world_T_cam_batch: Float[Tensor, "N 4 4"] = torch.stack(
        [torch.from_numpy(mv_pred.pinhole_param.extrinsics.world_T_cam) for mv_pred in multiview_predictions]
    ).float()

    # orient and center the poses
    output: tuple[Tensor, Tensor] = auto_orient_and_center_poses(
        poses=world_T_cam_batch, method="vertical", center_method="poses"
    )
    oriented_world_T_cam_batch: Float[Tensor, "N 3 4"] = output[0]
    bottom: Float[Tensor, "1 4"] = torch.Tensor([[0, 0, 0, 1]])
    bottom_batch: Float[Tensor, "N 1 4"] = repeat(bottom, "r c -> b r c", b=oriented_world_T_cam_batch.shape[0])
    oriented_world_T_cam_batch: Float[Tensor, "N 4 4"] = torch.cat([oriented_world_T_cam_batch, bottom_batch], dim=1)
    transform: Float[Tensor, "3 4"] = output[1]
    transform: Float[Tensor, "4 4"] = torch.cat([transform, bottom], dim=0)

    # set extrinsics to the oriented poses
    for idx, _ in enumerate(multiview_predictions):
        world_T_cam: Float[ndarray, "4 4"] = oriented_world_T_cam_batch[idx].numpy()
        multiview_predictions[idx].pinhole_param.extrinsics = Extrinsics(
            world_R_cam=world_T_cam[:3, :3], world_t_cam=world_T_cam[:3, 3]
        )

    # update point cloud based on newly oriented poses
    points = np.asarray(multiview_predictions[0].pointcloud.points)
    points = torch.from_numpy(points).float()
    points_homogeneous = torch.cat(
        [points, torch.ones(points.shape[0], 1)], dim=-1
    )  # Convert to homogeneous coordinates
    transformed_points = (transform @ points_homogeneous.T).T[:, :3]
    transformed_points = transformed_points.numpy().astype(np.float32)
    # update all point clouds

    # Create an empty point cloud
    transformed_pcd = o3d.geometry.PointCloud()

    # Ensure your positions and colors are of the appropriate type (typically float64 for points)
    transformed_pcd.points = o3d.utility.Vector3dVector(transformed_points)  # Scale to allow saving as uint16 later on
    transformed_pcd.colors = multiview_predictions[0].pointcloud.colors

    rr.log(
        f"{parent_log_path}/point_cloud",
        rr.Points3D(
            transformed_pcd.points,
            colors=transformed_pcd.colors,
        ),
        static=True,
    )
    mv_pred: MultiviewPred
    for mv_pred in multiview_predictions:
        cam_log_path: Path = parent_log_path / mv_pred.cam_name

        mask: Float32[ndarray, "H W"] = mv_pred.confidence_mask.astype(np.float32)
        depth_map: UInt16[ndarray, "H W"] = mv_pred.depth_map

        log_pinhole(
            mv_pred.pinhole_param,
            cam_log_path=cam_log_path,
            image_plane_distance=25.0,
        )

        rr.log(
            f"{cam_log_path}/pinhole/image",
            rr.Image(mv_pred.rgb_image, color_model=rr.ColorModel.RGB),
        )
        rr.log(
            f"{cam_log_path}/pinhole/confidence",
            rr.Image(mask),
        )
        rr.log(
            f"{cam_log_path}/pinhole/depth",
            rr.DepthImage(depth_map, draw_order=1),
        )

    # save the calibration data
    if config.output_dir is not None:
        save_to_nerfstudio(
            ns_save_path=config.output_dir,
            bgr_list=bgr_list,
            pinhole_param_list=[mv_pred.pinhole_param for mv_pred in multiview_predictions],
            pointcloud=transformed_pcd,
        )
