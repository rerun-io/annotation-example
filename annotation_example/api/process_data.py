import warnings
from dataclasses import dataclass
from pathlib import Path
from timeit import default_timer as timer
from typing import Literal, TypedDict, assert_never

import cv2
import numpy as np
import open3d as o3d
import rerun as rr
import rerun.blueprint as rrb
import torch
from jaxtyping import Bool, Float32, UInt8, UInt16
from monopriors.dc_utils import read_video_frames
from monopriors.depth_utils import depth_edges_mask, depth_to_points
from monopriors.multiview_models.vggt_model import MultiviewPred, VGGTPredictor
from monopriors.relative_depth_models.depth_anything_v2 import (
    DepthAnythingV2Predictor,
    RelativeDepthPrediction,
)
from monopriors.relative_depth_models.video_depth_anything import VideoDepthAnythingPredictor
from monopriors.scale_utils import compute_scale_and_shift
from numpy import ndarray
from serde.json import to_json
from simplecv.camera_parameters import PinholeParameters
from simplecv.rerun_log_utils import RerunTyroConfig, log_pinhole
from simplecv.video_io import MultiVideoReader
from tqdm import tqdm

# turn off the torch.cuda.amp.autocast FutureWarning
warnings.filterwarnings("ignore", category=FutureWarning, message=r".*torch\.cuda\.amp\.autocast.*")


def create_blueprint(parent_log_path: Path, image_paths: list[Path]) -> rrb.Blueprint:
    view3d = rrb.Spatial3DView(
        origin=f"{parent_log_path}",
        contents=[
            "+ $origin/**",
            # don't include depths in the 3D view, as they can be very noisy
            *[f"- /{parent_log_path}/camera_{i}/pinhole/depth" for i in range(len(image_paths))],
            *[f"- /{parent_log_path}/camera_{i}/pinhole/aligned_depth" for i in range(len(image_paths))],
        ],
    )
    view2d = rrb.Vertical(
        contents=[
            rrb.Horizontal(
                contents=[
                    # rrb.Spatial2DView(
                    #     origin=f"{parent_log_path}/camera_{i}/pinhole/",
                    #     contents=[
                    #         "+ $origin/**",
                    #     ],
                    #     name="Pinhole Content",
                    # ),
                    # rrb.Spatial2DView(
                    #     origin=f"{parent_log_path}/camera_{i}/pinhole/confidence",
                    #     contents=[
                    #         "+ $origin/**",
                    #     ],
                    #     name="Confidence Map",
                    # ),
                    rrb.Spatial2DView(
                        origin=f"{parent_log_path}/camera_{i}/pinhole/aligned_depth",
                        contents=[
                            "+ $origin/**",
                        ],
                        name="Aligned Depth",
                    ),
                    rrb.Spatial2DView(
                        origin=f"{parent_log_path}/camera_{i}/pinhole/image",
                        contents=[
                            "+ $origin/**",
                        ],
                        name="Image",
                    ),
                ]
            )
            # show at most 4 cameras
            for i in range(min(4, len(image_paths)))
        ]
    )

    blueprint = rrb.Blueprint(rrb.Horizontal(contents=[view3d, view2d], column_shares=[3, 1]), collapse_panels=True)
    return blueprint


@dataclass
class ProcessConfig:
    rr_config: RerunTyroConfig
    video_dir: Path = Path("/mnt/12tbdrive/data/HO-cap/sample/subject_8/20231024_180733/raw_videos/")
    device: Literal["cpu", "cuda"] = "cuda"
    confidence_threshold: float = 50.0
    depth_model: Literal["depthanythingv2", "videodepthanything", "vggt"] = "videodepthanything"
    max_video_len: int = -1
    output_dir: Path = Path("data/example_data")
    sequence_name: str = "0"
    viz_depth_videos: bool = False


def save_calibration_data(save_dir: Path, sequence_name: str, calibration_data: list[MultiviewPred]) -> Path:
    sequence_dir: Path = save_dir / sequence_name
    sequence_dir.mkdir(parents=True, exist_ok=True)
    # Create directories for videos, depth maps, and confidence maps
    videos_dir = sequence_dir / "videos"
    videos_dir.mkdir(parents=True, exist_ok=True)

    pinhole_parameters: list[PinholeParameters] = [calib_data.pinhole_param for calib_data in calibration_data]
    # Save camera parameters to a JSON file
    camera_parameters_path = sequence_dir / "camera_parameters.json"
    camera_parameters_json: str = to_json(pinhole_parameters)
    # Save the JSON string to a file
    with open(camera_parameters_path, "w") as f:
        f.write(camera_parameters_json)
    # Save point cloud to a PLY file
    point_cloud_path = sequence_dir / "point_cloud.ply"
    o3d.io.write_point_cloud(str(point_cloud_path), calibration_data[0].pointcloud)

    return sequence_dir


def process_data(config: ProcessConfig):
    parent_log_path: Path = Path("world")
    timeline_name = "frame_idx"
    video_paths = sorted(config.video_dir.glob("*.mp4"))
    assert len(video_paths) > 0, f"No videos found in {config.video_dir}"

    mv_reader = MultiVideoReader(video_paths=video_paths)

    blueprint = create_blueprint(parent_log_path=parent_log_path, image_paths=video_paths)
    rr.send_blueprint(blueprint)

    rr.set_time(timeline_name, sequence=0)

    bgr_list: list[UInt8[ndarray, "H W 3"]] = mv_reader[0]
    rgb_list: list[UInt8[ndarray, "H W 3"]] = [cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB) for bgr in bgr_list]

    device = "cuda" if torch.cuda.is_available() else "cpu"

    vggt_predictor = VGGTPredictor(
        device=device,
        confidence_threshold=config.confidence_threshold,
        preprocessing_mode="crop",
    )
    inference_start = timer()
    calibration_data: list[MultiviewPred] = vggt_predictor(rgb_list=rgb_list)
    print(f"Inference time: {timer() - inference_start:.2f} seconds")

    # rr.log(
    #     f"{parent_log_path}/point_cloud",
    #     rr.Points3D(
    #         calibration_data[0].pointcloud.points,
    #         colors=calibration_data[0].pointcloud.colors,
    #     ),
    #     # static=True,
    # )
    calib_data: MultiviewPred
    for calib_data in calibration_data:
        cam_log_path: Path = parent_log_path / calib_data.cam_name

        mask: Float32[ndarray, "H W"] = calib_data.confidence_mask.astype(np.float32)
        depth_map: UInt16[ndarray, "H W"] = calib_data.depth_map

        log_pinhole(
            calib_data.pinhole_param,
            cam_log_path=cam_log_path,
            image_plane_distance=100.0,
            # static=True,
        )

        # rr.log(
        #     f"{cam_log_path}/pinhole/image",
        #     rr.Image(calib_data.rgb_image, color_model=rr.ColorModel.RGB),
        #     # , static=True
        # )
        rr.log(
            f"{cam_log_path}/pinhole/confidence",
            rr.Image(mask),
            # static=True,
        )
        rr.log(
            f"{cam_log_path}/pinhole/depth",
            rr.DepthImage(depth_map, draw_order=1),
            # static=True,
        )

    # save the calibration data
    sequence_dir: Path = save_calibration_data(
        save_dir=config.output_dir, sequence_name=config.sequence_name, calibration_data=calibration_data
    )
    # # Clean up
    # torch.cuda.empty_cache()
    # del vggt_predictor

    # Define a typed dictionary for camera scale and shift
    class CameraScaleShift(TypedDict):
        scale: float
        shift: float

    start = timer()
    print("Generating Depth Maps...")
    aligned_depth_dict: dict[str, list[UInt16[np.ndarray, "H W"]]] = {}
    final_masks_dict: dict[str, list[UInt8[np.ndarray, "H W"]]] = {}
    match config.depth_model:
        case "depthanythingv2":
            DEPTH_PREDICTOR = DepthAnythingV2Predictor(device="cpu", encoder="vits")
            DEPTH_PREDICTOR.set_model_device("cuda")

            # Store scale and shift for each camera
            camera_scale_shift: dict[int, CameraScaleShift] = {}

            # propagate the prompts to get masklets throughout the video
            for frame_idx, bgr_list in tqdm(enumerate(mv_reader), desc="Processing frames", total=len(mv_reader)):
                rr.set_time(timeline_name, sequence=frame_idx)
                rgb_list: list[UInt8[ndarray, "H W 3"]] = [cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB) for bgr in bgr_list]

                for cam_idx, (rgb, calib_data) in enumerate(zip(rgb_list, calibration_data, strict=True)):
                    cam_log_path: Path = parent_log_path / calib_data.cam_name / "pinhole"
                    pinhole_param: PinholeParameters = calib_data.pinhole_param
                    depth_pred: RelativeDepthPrediction = DEPTH_PREDICTOR.__call__(
                        rgb=rgb, K_33=pinhole_param.intrinsics.k_matrix.astype(np.float32)
                    )

                    mono_disparity: Float32[np.ndarray, "h w"] = depth_pred.depth
                    mask = calib_data.confidence_mask.astype(np.bool_)
                    sparse_depth: Float32[ndarray, "H W"] = (calib_data.depth_map).astype(np.float32)

                    # Calculate scale and shift only on the first frame
                    if frame_idx == 0:
                        scale, shift = compute_scale_and_shift(
                            prediction=mono_disparity.astype(np.float32),
                            target=sparse_depth,
                            mask=mask,
                        )
                        camera_scale_shift[cam_idx] = {"scale": scale, "shift": shift}
                    else:
                        # Reuse scale and shift from first frame
                        scale: float = camera_scale_shift[cam_idx]["scale"]
                        shift: float = camera_scale_shift[cam_idx]["shift"]

                    # Calculate aligned depth
                    aligned_depth: Float32[np.ndarray, "h w"] = mono_disparity.astype(np.float32) * scale + shift

                    # Create a comprehensive mask combining all filtering conditions
                    final_mask = np.ones_like(aligned_depth, dtype=bool)

                    # Filter negative values
                    final_mask &= aligned_depth >= 0

                    # Filter values above max sparse depth
                    final_mask &= aligned_depth <= np.max(sparse_depth) * 0.8

                    # Filter depth edges
                    edges_mask: Bool[np.ndarray, "h w"] = depth_edges_mask(
                        aligned_depth, threshold=0.01 * 1000
                    )  # due to uint16 1000x scale up
                    final_mask &= ~edges_mask

                    # Apply the final mask
                    # Create a copy to avoid modifying the original aligned depth
                    aligned_masked_depth = aligned_depth.copy()
                    # Apply the final mask to the copy
                    aligned_masked_depth[~final_mask] = 0

                    # log to cam_log_path to avoid backprojecting disparity
                    if config.viz_depth_videos:
                        raw_pcd = depth_to_points(
                            depth_1hw=aligned_masked_depth.astype(np.float32)[np.newaxis, :, :],
                            K_33=calib_data.pinhole_param.intrinsics.k_matrix.astype(np.float32),
                            R=calib_data.pinhole_param.extrinsics.world_R_cam.astype(np.float32),
                            t=calib_data.pinhole_param.extrinsics.world_t_cam.astype(np.float32),
                        )

                        rr.log(f"{cam_log_path}/aligned_depth", rr.DepthImage(aligned_masked_depth))
                        rr.log(
                            f"{parent_log_path}/{calib_data.cam_name}_camera_pointcloud",
                            rr.Points3D(raw_pcd, colors=rgb.reshape(-1, 3)),
                        )
                        rr.log(
                            f"{cam_log_path}/image",
                            rr.Image(rgb, color_model=rr.ColorModel.RGB).compress(jpeg_quality=70),
                        )
        case "videodepthanything":
            DEPTH_PREDICTOR = VideoDepthAnythingPredictor(device="cuda", encoder="vits")
            # Store scale and shift for each camera
            camera_scale_shift: dict[int, CameraScaleShift] = {}
            # instead of iterating over the frames, we will iterate over the video reader
            for cam_idx, (video_path, calib_data) in enumerate(
                tqdm(
                    zip(mv_reader.video_paths, calibration_data, strict=True),
                    desc="Processing videos",
                    total=len(calibration_data),
                )
            ):
                cam_log_path: Path = parent_log_path / calib_data.cam_name / "pinhole"
                read_output: tuple[UInt8[ndarray, "T H W 3"], float] = read_video_frames(
                    video_path, process_length=config.max_video_len, target_fps=-1, max_res=-1
                )
                frames: UInt8[ndarray, "T H W 3"] = read_output[0]
                depths: list[RelativeDepthPrediction] = DEPTH_PREDICTOR(
                    frames, K_33=calib_data.pinhole_param.intrinsics.k_matrix.astype(np.float32)
                )
                aligned_depths_list: list[UInt16[np.ndarray, "H W"]] = []
                final_masks_list: list[UInt8[np.ndarray, "H W"]] = []
                for frame_idx, depth_pred in enumerate(depths):
                    rr.set_time(timeline_name, sequence=frame_idx)
                    mono_disparity: Float32[np.ndarray, "h w"] = depth_pred.depth
                    confidence_mask = calib_data.confidence_mask.astype(np.bool_)
                    sparse_depth: Float32[ndarray, "H W"] = calib_data.depth_map.astype(np.float32)

                    # Calculate scale and shift only on the first frame
                    if frame_idx == 0:
                        scale, shift = compute_scale_and_shift(
                            prediction=mono_disparity.astype(np.float32),
                            target=sparse_depth,
                            mask=confidence_mask,
                        )
                        camera_scale_shift[cam_idx] = {"scale": scale, "shift": shift}
                    else:
                        # Reuse scale and shift from first frame
                        scale: float = camera_scale_shift[cam_idx]["scale"]
                        shift: float = camera_scale_shift[cam_idx]["shift"]

                    # Calculate aligned depth
                    aligned_depth: Float32[np.ndarray, "h w"] = (mono_disparity * scale + shift).astype(np.float32)

                    # Create a comprehensive mask combining all filtering conditions
                    final_mask = np.ones_like(aligned_depth, dtype=bool)

                    # Filter negative values
                    final_mask &= aligned_depth >= 0

                    # Filter values above max sparse depth
                    final_mask &= aligned_depth <= np.max(sparse_depth) * 0.8

                    # Filter depth edges
                    edges_mask: Bool[np.ndarray, "h w"] = depth_edges_mask(
                        aligned_depth, threshold=0.01 * 1000
                    )  # due to uint16 1000x scale up
                    final_mask &= ~edges_mask

                    # convert to UInt8
                    save_mask: UInt8[ndarray, "h w"] = (final_mask.copy() * 255).astype(np.uint8)
                    final_masks_list.append(save_mask)
                    aligned_depths_list.append(aligned_depth)

                    # Apply the final mask
                    # Create a copy to avoid modifying the original aligned depth
                    aligned_masked_depth = aligned_depth.copy()
                    # Apply the final mask to the copy
                    aligned_masked_depth[~final_mask] = 0

                    # log to cam_log_path to avoid backprojecting disparity
                    if config.viz_depth_videos:
                        pointcloud = depth_to_points(
                            depth_1hw=aligned_masked_depth.astype(np.float32)[np.newaxis, :, :],
                            K_33=calib_data.pinhole_param.intrinsics.k_matrix.astype(np.float32),
                            R=calib_data.pinhole_param.extrinsics.world_R_cam.astype(np.float32),
                            t=calib_data.pinhole_param.extrinsics.world_t_cam.astype(np.float32),
                        )
                        rr.log(
                            f"{parent_log_path}/{calib_data.cam_name}_camera_pointcloud",
                            rr.Points3D(pointcloud, colors=frames[frame_idx].reshape(-1, 3)),
                        )
                        rr.log(f"{cam_log_path}/aligned_depth", rr.DepthImage(aligned_masked_depth))
                        rr.log(
                            f"{cam_log_path}/image",
                            rr.Image(frames[frame_idx], color_model=rr.ColorModel.RGB).compress(jpeg_quality=70),
                        )
                # add to dict
                aligned_depth_dict[calib_data.cam_name] = aligned_depths_list
                final_masks_dict[calib_data.cam_name] = final_masks_list
        case "vggt":
            # Store scale and shift for each camera
            camera_scale_shift: dict[int, CameraScaleShift] = {}
            for frame_idx, bgr_list in tqdm(enumerate(mv_reader), desc="Processing frames", total=len(mv_reader)):
                rr.set_time(timeline_name, sequence=frame_idx)
                rgb_list: list[UInt8[ndarray, "H W 3"]] = [cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB) for bgr in bgr_list]
                multiview_preds: list[MultiviewPred] = vggt_predictor(rgb_list=rgb_list)

                rr.log(
                    f"{parent_log_path}/point_cloud",
                    rr.Points3D(
                        multiview_preds[0].pointcloud.points,
                        colors=multiview_preds[0].pointcloud.colors,
                    ),
                )

                for cam_idx, (mv_pred, calib_data) in enumerate(zip(multiview_preds, calibration_data, strict=True)):
                    cam_log_path: Path = parent_log_path / calib_data.cam_name / "pinhole"

                    mask: Float32[ndarray, "H W"] = mv_pred.confidence_mask.astype(np.float32)
                    depth_map: UInt16[ndarray, "H W"] = mv_pred.depth_map

                    sparse_depth: Float32[ndarray, "H W"] = (calib_data.depth_map).astype(np.float32)

                    # Calculate scale and shift only on the first frame
                    if frame_idx == 0:
                        scale, shift = compute_scale_and_shift(
                            prediction=depth_map.astype(np.float32),
                            target=sparse_depth,
                            mask=mask.astype(np.bool_),
                        )
                        camera_scale_shift[cam_idx] = {"scale": scale, "shift": shift}
                    else:
                        # Reuse scale and shift from first frame
                        scale: float = camera_scale_shift[cam_idx]["scale"]
                        shift: float = camera_scale_shift[cam_idx]["shift"]

                    # Calculate aligned depth
                    aligned_depth: Float32[np.ndarray, "h w"] = depth_map.astype(np.float32) * scale + shift

                    # Create a comprehensive mask combining all filtering conditions
                    final_mask = np.ones_like(aligned_depth, dtype=bool)

                    # Filter negative values
                    final_mask &= aligned_depth >= 0

                    # Filter values above max sparse depth
                    final_mask &= aligned_depth <= np.max(sparse_depth) * 0.8

                    # Filter depth edges
                    edges_mask: Bool[np.ndarray, "h w"] = depth_edges_mask(
                        aligned_depth, threshold=0.01 * 1000
                    )  # due to uint16 1000x scale up
                    final_mask &= ~edges_mask

                    # Apply the final mask
                    # Create a copy to avoid modifying the original aligned depth
                    aligned_masked_depth = aligned_depth.copy()
                    # Apply the final mask to the copy
                    aligned_masked_depth[~final_mask] = 0

                    # log to cam_log_path to avoid backprojecting disparity
                    if config.viz_depth_videos:
                        rr.log(f"{cam_log_path}/aligned_depth", rr.DepthImage(depth_map))
                        rr.log(
                            f"{cam_log_path}/image",
                            rr.Image(rgb_list[cam_idx], color_model=rr.ColorModel.RGB).compress(jpeg_quality=70),
                        )
        case _:
            assert_never(config.depth_model)

    # assert len(aligned_depths_list) != 0, "No depth maps generated"
    # # Save the aligned depth maps and confidence masks
    # depth_dir: Path = sequence_dir / "aligned_depths"
    # depth_dir.mkdir(parents=True, exist_ok=True)
    # conf_dir: Path = sequence_dir / "confidence_masks"
    # conf_dir.mkdir(parents=True, exist_ok=True)
    # # Add tqdm progress bars for saving files
    # for cam_name, aligned_depths_list in tqdm(
    #     aligned_depth_dict.items(), desc="Saving depth maps by camera", total=len(aligned_depth_dict)
    # ):
    #     # Create camera specific directory
    #     (depth_dir / cam_name).mkdir(parents=True, exist_ok=True)
    #     for idx, depth in tqdm(
    #         enumerate(aligned_depths_list),
    #         desc=f"Saving depths for {cam_name}",
    #         total=len(aligned_depths_list),
    #         leave=False,
    #     ):
    #         # Convert to UInt16
    #         depth_path = depth_dir / cam_name / f"depth_{idx:06d}.png"
    #         # Save depth map as PNG
    #         cv2.imwrite(str(depth_path), depth.astype(np.uint16))
    #         # # eventually save as tiff when rerun supports it
    #         # depth_map_tiff_path = depth_dir / cam_name / "depth.tiff"
    #         # cv2.imwrite(str(depth_map_tiff_path), aligned_depths[0], [cv2.IMWRITE_TIFF_COMPRESSION, 8])
    # for cam_name, final_masks_list in tqdm(
    #     final_masks_dict.items(), desc="Saving confidence masks by camera", total=len(final_masks_dict)
    # ):
    #     # Create camera specific directory
    #     (conf_dir / cam_name).mkdir(parents=True, exist_ok=True)
    #     for idx, final_mask in tqdm(
    #         enumerate(final_masks_list),
    #         desc=f"Saving masks for {cam_name}",
    #         total=len(final_masks_list),
    #         leave=False,
    #     ):
    #         # Convert to UInt16
    #         mask_path = conf_dir / cam_name / f"conf_{idx:06d}.png"
    #         cv2.imwrite(
    #             str(mask_path),
    #             final_mask,
    #         )
    elapsed = timer() - start
    mins, secs = divmod(elapsed, 60)
    print(f"Depth Maps from {config.depth_model} generated in {int(mins)}m {secs:.2f}s")
