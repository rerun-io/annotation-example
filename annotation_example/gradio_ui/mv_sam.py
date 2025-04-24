import gc
import shutil
import uuid
from pathlib import Path
from typing import Literal, assert_never, no_type_check

import cv2
import gradio as gr
import numpy as np
import open3d as o3d
import rerun as rr
import rerun.blueprint as rrb
import torch
from einops import rearrange
from gradio_rerun import Rerun
from jaxtyping import Bool, Float, Float32, Int, UInt8, UInt16
from sam2.sam2_image_predictor import SAM2ImagePredictor
from sam2.sam2_video_predictor import SAM2VideoPredictor
from simplecv.camera_parameters import PinholeParameters
from simplecv.conversion_utils import save_to_nerfstudio
from simplecv.data.exoego.assembly_101 import Assembly101Sequence
from simplecv.data.exoego.hocap import ExoCameraIDs, HOCapSequence
from simplecv.ops.triangulate import batch_triangulate, projectN3
from simplecv.ops.tsdf_depth_fuser import Open3DFuser
from simplecv.video_io import MultiVideoReader, VideoReader
from tqdm import tqdm

from annotation_example.gradio_ui.mv_sam_callbacks import (
    KeypointsContainer,
    RerunLogPaths,
    get_recording,
    update_keypoints,
)

if gr.NO_RELOAD:
    VIDEO_SAM_PREDICTOR: SAM2VideoPredictor = SAM2VideoPredictor.from_pretrained("facebook/sam2-hiera-tiny")
    IMG_SAM_PREDICTOR: SAM2ImagePredictor = SAM2ImagePredictor.from_pretrained("facebook/sam2-hiera-tiny")


def create_blueprint(
    exo_video_log_paths: list[Path], num_videos_to_log: Literal[4, 8] = 8, active_tab: int = 0
) -> rrb.Blueprint:
    main_view = rrb.Vertical(
        contents=[
            rrb.Spatial3DView(
                origin="/",
            ),
            # take the first 4 video files
            rrb.Horizontal(
                contents=[
                    rrb.Tabs(
                        rrb.Spatial2DView(origin=f"{video_log_path.parent}"),
                        rrb.Spatial2DView(
                            origin=f"{video_log_path}".replace("image", "video"),
                        ),
                        active_tab=active_tab,
                    )
                    for video_log_path in exo_video_log_paths[:4]
                ]
            ),
        ],
        row_shares=[3, 1],
    )
    additional_views = rrb.Vertical(
        contents=[
            rrb.Tabs(
                rrb.Spatial2DView(origin=f"{video_log_path.parent}"),
                rrb.Spatial2DView(origin=f"{video_log_path}".replace("image", "video")),
                active_tab=active_tab,
            )
            for video_log_path in exo_video_log_paths[4:]
        ]
    )
    # do the last 4 videos
    contents = [main_view]
    if num_videos_to_log == 8:
        contents.append(additional_views)

    blueprint = rrb.Blueprint(
        rrb.Horizontal(
            contents=contents,
            column_shares=[4, 1],
        ),
        collapse_panels=True,
    )
    return blueprint


def log_pinhole_rec(
    rec: rr.RecordingStream,
    camera: PinholeParameters,
    cam_log_path: Path,
    image_plane_distance: float = 0.5,
    static: bool = False,
) -> None:
    """
    Logs the pinhole camera parameters and transformation data.

    Parameters:
    camera (PinholeParameters): The pinhole camera parameters including intrinsics and extrinsics.
    cam_log_path (Path): The path where the camera log will be saved.
    image_plane_distance (float, optional): The distance of the image plane from the camera. Defaults to 0.5.
    static (bool, optional): If True, the log data will be marked as static. Defaults to False.

    Returns:
    None
    """
    # camera intrinsics
    rec.log(
        f"{cam_log_path}/pinhole",
        rr.Pinhole(
            image_from_camera=camera.intrinsics.k_matrix,
            height=camera.intrinsics.height,
            width=camera.intrinsics.width,
            camera_xyz=getattr(
                rr.ViewCoordinates,
                camera.intrinsics.camera_conventions,
            ),
            image_plane_distance=image_plane_distance,
        ),
        static=static,
    )
    # camera extrinsics
    rec.log(
        f"{cam_log_path}",
        rr.Transform3D(
            translation=camera.extrinsics.cam_t_world,
            mat3x3=camera.extrinsics.cam_R_world,
            from_parent=True,
        ),
        static=static,
    )


def rescale_img(img_hw3: UInt8[np.ndarray, "h w 3"], max_dim: int) -> UInt8[np.ndarray, "... 3"]:
    # resize the image to have a max dim of max_dim
    height, width, _ = img_hw3.shape
    current_dim = max(height, width)

    # If current dimension is larger than max_dim, calculate scale factor
    if current_dim > max_dim:
        scale_factor = max_dim / current_dim
        new_height = int(height * scale_factor)
        new_width = int(width * scale_factor)

        # Resize image maintaining aspect ratio
        resized_img = cv2.resize(img_hw3, (new_width, new_height), interpolation=cv2.INTER_AREA)
        return resized_img

    # Return original image if no resize needed
    return img_hw3


@no_type_check
def reset_keypoints(
    active_recording_id: uuid.UUID, mv_keypoint_dict: dict[str, KeypointsContainer], log_paths: RerunLogPaths
):
    yield from _reset_keypoints(
        active_recording_id=active_recording_id,
        mv_keypoint_dict=mv_keypoint_dict,
        log_paths=log_paths,
    )


def _reset_keypoints(
    active_recording_id: uuid.UUID, mv_keypoint_dict: dict[str, KeypointsContainer], log_paths: RerunLogPaths
):
    # Now we can produce a valid keypoint.
    rec: rr.RecordingStream = get_recording(active_recording_id)
    stream: rr.BinaryStream = rec.binary_stream()

    mv_keypoint_dict: dict[str, KeypointsContainer] = {
        cam_name: KeypointsContainer.empty() for cam_name in mv_keypoint_dict
    }

    rec.set_time(log_paths["timeline_name"], sequence=0)
    # Log include points if any exist
    for cam_log_path in log_paths["cam_log_path_list"]:
        pinhole_path: Path = cam_log_path / "pinhole"
        rec.log(
            f"{pinhole_path}/image/include",
            rr.Clear(recursive=True),
        )
        rec.log(
            f"{pinhole_path}/image/exclude",
            rr.Clear(recursive=True),
        )
        rec.log(
            f"{pinhole_path}/image/bbox",
            rr.Clear(recursive=True),
        )
        rec.log(
            f"{pinhole_path}/image/bbox_center",
            rr.Clear(recursive=True),
        )
        rec.log(
            f"{pinhole_path}/segmentation",
            rr.Clear(recursive=True),
        )
        rec.log(
            f"{pinhole_path}/depth",
            rr.Clear(recursive=True),
        )

    rec.log(
        f"{log_paths['parent_log_path']}/triangulated",
        rr.Clear(recursive=True),
    )

    # Ensure we consume everything from the recording.
    stream.flush()
    yield stream.read(), mv_keypoint_dict, {}


@no_type_check
def get_initial_mask(
    recording_id: uuid.UUID,
    mv_keypoints_dict: dict[str, KeypointsContainer],
    log_paths: RerunLogPaths,
    rgb_list: list[UInt8[np.ndarray, "h w 3"]],
    keypoint_centers_dict: dict[str, Float32[np.ndarray, "3"]],
):
    yield from _get_initial_mask(
        recording_id=recording_id,
        mv_keypoints_dict=mv_keypoints_dict,
        log_paths=log_paths,
        rgb_list=rgb_list,
        keypoint_centers_dict=keypoint_centers_dict,
    )


def _get_initial_mask(
    recording_id: uuid.UUID,
    mv_keypoints_dict: dict[str, KeypointsContainer],
    log_paths: RerunLogPaths,
    rgb_list: list[UInt8[np.ndarray, "h w 3"]],
    keypoint_centers_dict: dict[str, Float[np.ndarray, "3"]],
):
    rec = get_recording(recording_id)
    stream = rec.binary_stream()

    rec.set_time(log_paths["timeline_name"], sequence=0)

    for (cam_name, keypoint_container), rgb in zip(mv_keypoints_dict.items(), rgb_list, strict=True):
        IMG_SAM_PREDICTOR.set_image(rgb)
        pinhole_log_path: Path = log_paths["parent_log_path"] / cam_name / "pinhole"
        points: Float32[np.ndarray, "num_points 2"] = np.vstack(
            [keypoint_container.include_points, keypoint_container.exclude_points]
        ).astype(np.float32)
        if points.shape[0] == 0:
            IMG_SAM_PREDICTOR.reset_predictor()
            rec.log(
                "logs",
                rr.TextLog("No points selected, skipping segmentation.", level="info"),
            )
        else:
            # Create labels array: 1 for include points, 0 for exclude points
            labels: Int[np.ndarray, "num_points"] = np.ones(len(keypoint_container.include_points), dtype=np.int32)  # noqa: UP037
            if len(keypoint_container.exclude_points) > 0:
                labels = np.concatenate([labels, np.zeros(len(keypoint_container.exclude_points), dtype=np.int32)])

            with torch.inference_mode():
                masks, scores, _ = IMG_SAM_PREDICTOR.predict(
                    point_coords=points,
                    point_labels=labels,
                    multimask_output=False,
                )
                masks: Bool[np.ndarray, "1 h w"] = masks > 0.0

            rec.log(
                f"{pinhole_log_path}/segmentation",
                rr.SegmentationImage(masks[0].astype(np.uint8)),
            )
            # Convert the mask to a bounding box
            if masks[0].any():
                y_min, y_max = np.where(masks[0].any(axis=1))[0][[0, -1]]
                x_min, x_max = np.where(masks[0].any(axis=0))[0][[0, -1]]
                bbox = np.array([x_min, y_min, x_max, y_max], dtype=np.float32)
                rec.log(
                    f"{pinhole_log_path}/video/bbox",
                    rr.Boxes2D(array=bbox, array_format=rr.Box2DFormat.XYXY, colors=(0, 0, 255)),
                )

                # Calculate the center of the bounding box
                center_xyc: Float32[np.ndarray, "3"] = np.array(  # noqa: UP037
                    [(x_min + x_max) / 2, (y_min + y_max) / 2, 1], dtype=np.float32
                )
                rec.log(
                    f"{pinhole_log_path}/video/bbox_center",
                    rr.Points2D(positions=(center_xyc[0], center_xyc[1]), colors=(0, 0, 255), radii=5),
                )
                keypoint_centers_dict[cam_name] = center_xyc
            IMG_SAM_PREDICTOR.reset_predictor()

        yield stream.read(), keypoint_centers_dict


@no_type_check
def triangulate_centers(
    recording_id: uuid.UUID,
    center_xyc_dict: dict[str, Float32[np.ndarray, "3"]],
    exo_cam_list: list[PinholeParameters],
    log_paths: RerunLogPaths,
    rgb_list: list[UInt8[np.ndarray, "h w 3"]],
):
    yield from _triangulate_centers(
        recording_id=recording_id,
        center_xyc_dict=center_xyc_dict,
        exo_cam_list=exo_cam_list,
        log_paths=log_paths,
        rgb_list=rgb_list,
    )


def _triangulate_centers(
    recording_id: uuid.UUID,
    center_xyc_dict: dict[str, Float32[np.ndarray, "3"]],
    exo_cam_list: list[PinholeParameters],
    log_paths: RerunLogPaths,
    rgb_list: list[UInt8[np.ndarray, "h w 3"]],
):
    rec = get_recording(recording_id)
    stream = rec.binary_stream()

    masks_list: list[UInt8[np.ndarray, "h w"]] = []
    new_center_xyc_dict: dict[str, Float32[np.ndarray, "3"]] = {}  # noqa: UP037
    button_visibility_update = gr.Button(visible=False)

    rec.set_time(log_paths["timeline_name"], sequence=0)
    if len(center_xyc_dict) >= 2:
        centers_xyc: Float32[np.ndarray, "num_views 3"] = np.stack(
            [center_xyc for center_xyc in center_xyc_dict.values() if center_xyc is not None], axis=0
        ).astype(np.float32)

        centers_xyc = rearrange(centers_xyc, "num_views xyc -> num_views 1 xyc")
        proj_matrices: list[Float32[np.ndarray, "3 4"]] = [
            exo_cam.projection_matrix.astype(np.float32) for exo_cam in exo_cam_list
        ]
        proj_matrices: Float32[np.ndarray, "num_views 3 4"] = np.stack(proj_matrices, axis=0).astype(np.float32)

        proj_matrices_filtered: list[Float32[np.ndarray, "3 4"]] = [
            exo_cam.projection_matrix.astype(np.float32) for exo_cam in exo_cam_list if exo_cam.name in center_xyc_dict
        ]
        proj_matrices_filtered: Float32[np.ndarray, "num_views 3 4"] = np.stack(proj_matrices_filtered, axis=0).astype(
            np.float32
        )
        xyzc: Float[np.ndarray, "n_points 4"] = batch_triangulate(
            keypoints_2d=centers_xyc, projection_matrices=proj_matrices_filtered
        )
        rec.log(
            f"{log_paths['parent_log_path']}/triangulated", rr.Points3D(xyzc[:, 0:3], colors=(0, 0, 255), radii=0.1)
        )

        projected_xyc: Float[np.ndarray, "num_views 1 3"] = projectN3(
            xyzc,
            proj_matrices,
        )

        xyc: Float32[np.ndarray, "1 3"]
        for rgb, cam_log_path, xyc in zip(rgb_list, log_paths["cam_log_path_list"], projected_xyc, strict=True):
            pinhole_log_path: Path = cam_log_path / "pinhole"
            # # append the new center to the dictionary
            new_center_xyc_dict[cam_log_path.name] = xyc.squeeze(0)
            xy = xyc[:, 0:2]
            rec.log(
                f"{pinhole_log_path}/video/bbox_center",
                rr.Points2D(positions=xy, colors=(0, 0, 255), radii=5),
            )
            IMG_SAM_PREDICTOR.set_image(rgb)
            labels: Int[np.ndarray, "num_points"] = np.ones(len(xyc), dtype=np.int32)  # noqa: UP037
            with torch.inference_mode():
                masks, scores, _ = IMG_SAM_PREDICTOR.predict(
                    point_coords=xy,
                    point_labels=labels,
                    multimask_output=False,
                )
                masks: Bool[np.ndarray, "1 h w"] = masks > 0.0

            mask = masks[0].astype(np.uint8)
            masks_list.append(mask)
            rec.log(
                f"{pinhole_log_path}/segmentation",
                rr.SegmentationImage(mask),
            )
            if mask.any():
                y_min, y_max = np.where(masks[0].any(axis=1))[0][[0, -1]]
                x_min, x_max = np.where(masks[0].any(axis=0))[0][[0, -1]]
                bbox = np.array([x_min, y_min, x_max, y_max], dtype=np.float32)
                rec.log(
                    f"{pinhole_log_path}/video/bbox",
                    rr.Boxes2D(array=bbox, array_format=rr.Box2DFormat.XYXY, colors=(0, 0, 255)),
                )

                # Calculate the center of the bounding box
                center_xyc: Float32[np.ndarray, "3"] = np.array(  # noqa: UP037
                    [(x_min + x_max) / 2, (y_min + y_max) / 2, 1], dtype=np.float32
                )
                rec.log(
                    f"{pinhole_log_path}/video/bbox_center",
                    rr.Points2D(positions=(center_xyc[0], center_xyc[1]), colors=(0, 0, 255), radii=5),
                )
            IMG_SAM_PREDICTOR.reset_predictor()

        button_visibility_update = gr.Button(visible=True)

    else:
        rec.log(
            "logs",
            rr.TextLog("No points selected, skipping segmentation.", level="info"),
        )
        new_center_xyc_dict = center_xyc_dict
        gr.Info("Not enough points to triangulate.")

    yield stream.read(), masks_list, new_center_xyc_dict, button_visibility_update


@no_type_check
def log_dataset(dataset_name: Literal["hocap", "assembly101"]):
    yield from _log_dataset(dataset_name)


def _log_dataset(dataset_name: Literal["hocap", "assembly101"]):
    recording_id: uuid.UUID = uuid.uuid4()
    rec: rr.RecordingStream = get_recording(recording_id)
    stream: rr.BinaryStream = rec.binary_stream()

    match dataset_name:
        case "hocap":
            sequence: HOCapSequence = HOCapSequence(
                data_path=Path("data/hocap/sample"),
                sequence_name="20231024_180733",
                subject_id="8",
                load_labels=False,
            )
        case "assembly101":
            # raise NotImplementedError("Assembly101 is not implemented yet.")
            sequence: Assembly101Sequence = Assembly101Sequence(
                data_path=Path("data/assembly101-sample"),
                sequence_name="nusar-2021_action_both_9015-b05b_9015_user_id_2021-02-02_161800",
                subject_id=None,
                load_labels=False,
            )
        case _:
            assert_never(dataset_name)

    parent_log_path: Path = Path("world")
    timeline_name: str = "frame_idx"

    rec.set_time(timeline_name, sequence=0)

    exo_video_readers: MultiVideoReader = sequence.exo_video_readers
    exo_video_files: list[Path] = exo_video_readers.video_paths
    exo_cam_log_paths: list[Path] = [parent_log_path / exo_cam.name for exo_cam in sequence.exo_cam_list]
    exo_video_log_paths: list[Path] = [cam_log_paths / "pinhole" / "image" for cam_log_paths in exo_cam_log_paths]

    initial_blueprint = create_blueprint(exo_video_log_paths, num_videos_to_log=8)
    rec.send_blueprint(initial_blueprint)
    rec.log("/", sequence.world_coordinate_system, static=True)

    bgr_list: list[UInt8[np.ndarray, "h w 3"]] = exo_video_readers[0]
    rgb_list: list[UInt8[np.ndarray, "h w 3"]] = [cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB) for bgr in bgr_list]
    # check if depth images exist
    if not sequence.depth_paths:
        depth_paths = None
    else:
        depth_paths: dict[ExoCameraIDs, Path] = sequence.depth_paths[0]
    exo_cam_list: list[PinholeParameters] = sequence.exo_cam_list

    cam_log_path_list: list[Path] = []
    fuser = Open3DFuser(fusion_resolution=0.02, max_fusion_depth=1.25)
    # log stationary exo cameras and video assets
    for exo_cam in exo_cam_list:
        cam_log_path: Path = parent_log_path / exo_cam.name
        cam_log_path_list.append(cam_log_path)
        image_plane_distance: float = 0.1 if dataset_name == "hocap" else 100.0
        log_pinhole_rec(
            rec=rec,
            camera=exo_cam,
            cam_log_path=cam_log_path,
            image_plane_distance=image_plane_distance,
            static=True,
        )

    for rgb, cam_log_path, exo_cam in zip(rgb_list, cam_log_path_list, exo_cam_list, strict=True):
        pinhole_log_path: Path = cam_log_path / "pinhole"
        rec.log(f"{pinhole_log_path}/image", rr.Image(rgb, color_model=rr.ColorModel.RGB), static=True)
        # rec.log(f"{pinhole_log_path}/depth", rr.DepthImage(depth_image, meter=1000))
        if depth_paths is not None:
            depth_path: Path = depth_paths[cam_log_path.name]
            depth_image: UInt16[np.ndarray, "480 640"] = cv2.imread(str(depth_path), cv2.IMREAD_ANYDEPTH)
            fuser.fuse_frames(
                depth_image,
                exo_cam.intrinsics.k_matrix,
                exo_cam.extrinsics.cam_T_world,
                rgb,
            )

    if depth_paths is not None:
        mesh: o3d.geometry.TriangleMesh = fuser.get_mesh()
        mesh.compute_vertex_normals()

        rec.log(
            f"{parent_log_path}/mesh",
            rr.Mesh3D(
                vertex_positions=mesh.vertices,
                triangle_indices=mesh.triangles,
                vertex_normals=mesh.vertex_normals,
                vertex_colors=mesh.vertex_colors,
            ),
            # static=True,
        )

        pcd: o3d.geometry.PointCloud = mesh.sample_points_poisson_disk(
            number_of_points=20_000,
        )

    log_paths = RerunLogPaths(
        timeline_name=timeline_name,
        parent_log_path=parent_log_path,
        cam_log_path_list=cam_log_path_list,
    )

    mv_keypoint_dict: dict[str, KeypointsContainer] = {
        cam_log_path.name: KeypointsContainer.empty() for cam_log_path in cam_log_path_list
    }

    yield stream.read(), recording_id, log_paths, mv_keypoint_dict, rgb_list, exo_cam_list, pcd, exo_video_files


def handle_export(
    exo_cam_list: list[PinholeParameters],
    rgb_list: list[UInt8[np.ndarray, "h w 3"]],
    masks_list: list[UInt8[np.ndarray, "h w"]],
    pointcloud: o3d.geometry.PointCloud,
):
    bgr_list: list[UInt8[np.ndarray, "h w 3"]] = [cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR) for rgb in rgb_list]
    ns_save_dir: Path = Path("data/nerfstudio-export")
    masks_list: list[UInt8[np.ndarray, "h w"]] | None = masks_list if len(masks_list) > 0 else None
    save_to_nerfstudio(
        ns_save_path=ns_save_dir,
        pinhole_param_list=exo_cam_list,
        bgr_list=bgr_list,
        pointcloud=pointcloud,
        masks_list=masks_list,
    )

    # Define the path for the output zip file
    zip_output_path = Path("data/nerfstudio-output")
    zip_file_path: str = shutil.make_archive(str(zip_output_path), "zip", str(ns_save_dir))

    # Return the path to the zip file and switch tabs
    return gr.Tabs(selected=1), zip_file_path


@no_type_check
def propagate_mask(
    recording_id: uuid.UUID,
    dataset_name: Literal["hocap", "assembly101"],
    center_xyc_dict: dict[str, Float32[np.ndarray, "3"]],
    video_files: list[Path],
    log_paths: RerunLogPaths,
):
    yield from _propagate_mask(recording_id, dataset_name, center_xyc_dict, video_files, log_paths)


def _propagate_mask(
    recording_id: uuid.UUID,
    dataset_name: Literal["hocap", "assembly101"],
    center_xyc_dict: dict[str, Float[np.ndarray, "3"]],
    video_files: list[Path],
    log_paths: RerunLogPaths,
):
    rec = get_recording(recording_id)
    stream = rec.binary_stream()

    ## reload dataset to get depth paths for tsdf fusion later one
    match dataset_name:
        case "hocap":
            sequence: HOCapSequence = HOCapSequence(
                data_path=Path("data/hocap/sample"),
                sequence_name="20231024_180733",
                subject_id="8",
                load_labels=False,
            )
        case "assembly101":
            # raise NotImplementedError("Assembly101 is not implemented yet.")
            sequence: Assembly101Sequence = Assembly101Sequence(
                data_path=Path("data/assembly101-sample"),
                sequence_name="nusar-2021_action_both_9015-b05b_9015_user_id_2021-02-02_161800",
                subject_id=None,
                load_labels=False,
            )
        case _:
            assert_never(dataset_name)

    exo_cam_log_paths: list[Path] = [log_paths["parent_log_path"] / exo_cam.name for exo_cam in sequence.exo_cam_list]
    exo_video_log_paths: list[Path] = [cam_log_paths / "pinhole" / "video" for cam_log_paths in exo_cam_log_paths]

    new_blueprint = create_blueprint(exo_video_log_paths=exo_video_log_paths, num_videos_to_log=8, active_tab=1)
    rec.send_blueprint(new_blueprint)

    masks_dict: dict[str, list[UInt8[np.ndarray, "h w"]]] = {}

    # get video masks
    for video_path in video_files:  # Renamed for clarity
        video_reader: VideoReader = VideoReader(video_path)
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            inference_state = VIDEO_SAM_PREDICTOR.init_state(video_path=str(video_path))

            cam_name: str = video_path.parent.name
            pinhole_log_path: Path = log_paths["parent_log_path"] / cam_name / "pinhole"
            rec.set_time(log_paths["timeline_name"], sequence=0)
            rec.log(
                f"{pinhole_log_path}",
                rr.Clear(recursive=True),
            )

            points: Float[np.ndarray, "2"] = center_xyc_dict[cam_name][0:2]  # noqa: UP037
            points: Float32[np.ndarray, "num_points 2"] = rearrange(points, "uv -> 1 uv").astype(np.float32)
            labels: Int[np.ndarray, "num_points"] = np.array([1], dtype=np.int32)  # noqa: UP037

            frame_idx, object_ids, masks = VIDEO_SAM_PREDICTOR.add_new_points_or_box(
                inference_state, frame_idx=0, obj_id=0, points=points, labels=labels
            )

            # Get total number of frames. This might depend on how VIDEO_SAM_PREDICTOR stores it.
            # Assuming it's accessible via inference_state or the predictor object.
            # If not available directly, might need to read video metadata separately.
            try:
                # Attempt to get frame count, adjust key if necessary based on predictor implementation
                total_frames = inference_state["num_frames"]
                stop_frame = total_frames // 6  # Calculate one-third of the frames
            except KeyError:
                print(f"Warning: Could not determine total frames for {video_path}. Processing all frames.")
                stop_frame = float("inf")  # Process all frames if count is unknown

            frame_idx: int
            object_ids: list
            masks: Float32[torch.Tensor, "b 3 h w"]

            masks_list: list[UInt8[np.ndarray, "h w"]] = []
            # propagate the prompts to get masklets throughout the video
            for frame_idx, object_ids, masks in VIDEO_SAM_PREDICTOR.propagate_in_video(inference_state):  # noqa: B007
                # Only process and log up to the midpoint frame
                if frame_idx < stop_frame:
                    rec.set_time(log_paths["timeline_name"], sequence=frame_idx)
                    # Assuming masks[0] corresponds to the object of interest
                    masks: Bool[np.ndarray, "1 h w"] = (masks[0] > 0.0).numpy(force=True)
                    masks_list.append(masks[0].astype(np.uint8))

                    rec.log(
                        f"{pinhole_log_path}/video/segmentation",
                        rr.SegmentationImage(masks[0].astype(np.uint8)),
                    )

                    rec.log(
                        f"{pinhole_log_path}/video",
                        rr.Image(video_reader[frame_idx], color_model=rr.ColorModel.BGR).compress(jpeg_quality=5),
                    )

                    yield stream.read()
                else:
                    # Stop processing this video once the midpoint is reached
                    print(f"Reached midpoint frame {stop_frame} for {video_path}. Stopping propagation logging.")
                    break

            masks_dict[cam_name] = masks_list
            VIDEO_SAM_PREDICTOR.reset_state(inference_state)
            del inference_state
            gc.collect()
            torch.cuda.empty_cache()

    # Fuse depth images
    fuser = Open3DFuser(fusion_resolution=0.02, max_fusion_depth=1.25)
    rec.log(f"{log_paths['parent_log_path']}/triangulated", rr.Clear(recursive=True))
    yield stream.read()

    depth_paths: list[dict[str, Path]] | None = sequence.depth_paths
    if depth_paths is not None:
        for idx, depths_dict in enumerate(tqdm(depth_paths, desc="Logging depth images")):
            if idx >= stop_frame:
                break
            rec.set_time(timeline=log_paths["timeline_name"], sequence=idx)
            fuser = Open3DFuser(fusion_resolution=0.01, max_fusion_depth=1.25)
            bgr_list = sequence.exo_video_readers[idx]
            rec.log(f"{log_paths['parent_log_path']}/triangulated", rr.Clear(recursive=True))
            for exo_cam, bgr in zip(sequence.exo_cam_list, bgr_list, strict=True):
                depth_path = depths_dict[exo_cam.name]
                assert depth_path.exists(), f"Path {depth_path} does not exist."
                depth_image: UInt16[np.ndarray, "480 640"] = cv2.imread(str(depth_path), cv2.IMREAD_ANYDEPTH)
                rgb_hw3 = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                masks: UInt8[np.ndarray, "h w"] = masks_dict[exo_cam.name][idx]
                # update rgb_hw3 with the mask so that the fuser can use it, set the color to blue where mask is 1
                rgb_hw3[masks == 1] = [0, 0, 255]
                fuser.fuse_frames(
                    depth_image,
                    exo_cam.intrinsics.k_matrix,
                    exo_cam.extrinsics.cam_T_world,
                    rgb_hw3,
                )
            mesh: o3d.geometry.TriangleMesh = fuser.get_mesh()
            mesh.compute_vertex_normals()

            rec.log(
                f"{log_paths['parent_log_path']}/mesh",
                rr.Mesh3D(
                    vertex_positions=mesh.vertices,
                    triangle_indices=mesh.triangles,
                    vertex_normals=mesh.vertex_normals,
                    vertex_colors=mesh.vertex_colors,
                ),
                # static=True,
            )
            yield stream.read()
    else:
        print("No depth images found.")


with gr.Blocks() as mv_sam_block:
    mv_keypoint_dict: dict[str, KeypointsContainer] | gr.State = gr.State({})
    rgb_list: list[UInt8[np.ndarray, "h w 3"]] | gr.State = gr.State()
    masks_list: list[UInt8[np.ndarray, "h w"]] | gr.State = gr.State([])
    exo_cam_list: list[PinholeParameters] | gr.State = gr.State([])
    pointcloud: o3d.geometry.PointCloud | gr.State = gr.State()
    centers_xyc_dict: dict[str, Float32[np.ndarray, "3"]] | gr.State = gr.State({})
    video_files: list[Path] | gr.State = gr.State([])

    with gr.Row():
        with gr.Tabs() as main_tabs:
            with gr.TabItem("Controls", id=0):
                with gr.Column(scale=1):
                    dataset_dropdown = gr.Dropdown(
                        label="Dataset",
                        choices=["hocap"],  # TODO add assembly101
                        value="hocap",
                    )
                    load_dataset_btn = gr.Button("Load Dataset")

                    point_type = gr.Radio(
                        label="point type",
                        choices=["include", "exclude"],
                        value="include",
                        scale=1,
                    )
                    clear_points_btn = gr.Button("Clear Points", scale=1)
                    get_initial_mask_btn = gr.Button("Get Initial Mask", scale=1)
                    triangulate_btn = gr.Button("Triangulate Center", scale=1)
                    propagate_masks_btn = gr.Button("Propagate Masks", visible=False, scale=1)
                    export_btn = gr.Button("Export", scale=1)
            with gr.TabItem("Output", id=1):
                gr.Markdown("here you can see the output of the selected video")
                output_zip = gr.File(label="Exported Zip File", file_count="single", type="filepath")
        with gr.Column(scale=4):
            viewer = Rerun(
                streaming=True,
                panel_states={
                    "time": "collapsed",
                    "blueprint": "hidden",
                    "selection": "hidden",
                },
                height=700,
            )

    # We make a new recording id, and store it in a Gradio's session state.
    recording_id = gr.State()
    log_paths = gr.State({})

    load_dataset_btn.click(
        fn=log_dataset,
        inputs=[dataset_dropdown],
        outputs=[
            viewer,
            recording_id,
            log_paths,
            mv_keypoint_dict,
            rgb_list,
            exo_cam_list,
            pointcloud,
            video_files,
        ],
    )

    viewer.selection_change(
        update_keypoints,
        inputs=[
            recording_id,
            point_type,
            mv_keypoint_dict,
            log_paths,
        ],
        outputs=[viewer, mv_keypoint_dict],
    )

    clear_points_btn.click(
        fn=reset_keypoints,
        inputs=[recording_id, mv_keypoint_dict, log_paths],
        outputs=[viewer, mv_keypoint_dict, centers_xyc_dict],
    )

    get_initial_mask_btn.click(
        fn=get_initial_mask,
        inputs=[recording_id, mv_keypoint_dict, log_paths, rgb_list, centers_xyc_dict],
        outputs=[viewer, centers_xyc_dict],
    )

    triangulate_btn.click(
        fn=triangulate_centers,
        inputs=[recording_id, centers_xyc_dict, exo_cam_list, log_paths, rgb_list],
        outputs=[viewer, masks_list, centers_xyc_dict, propagate_masks_btn],
    )
    # TODO export masks + ply + camera poses for use with brush
    export_btn.click(
        fn=handle_export,
        inputs=[exo_cam_list, rgb_list, masks_list, pointcloud],
        outputs=[main_tabs, output_zip],
    )

    propagate_masks_btn.click(
        fn=propagate_mask,
        inputs=[recording_id, dataset_dropdown, centers_xyc_dict, video_files, log_paths],
        outputs=[viewer],
    )
