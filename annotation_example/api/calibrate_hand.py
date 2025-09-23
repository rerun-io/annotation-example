from dataclasses import dataclass
from pathlib import Path
from timeit import default_timer as timer
from typing import Literal, NamedTuple

import cv2
import numpy as np
import open3d as o3d
import rerun as rr
import rerun.blueprint as rrb
import torch
from jaxopt._src.levenberg_marquardt import LevenbergMarquardtState
from jaxtyping import Float, Float32, Int, UInt8
from natsort import natsorted
from numpy import ndarray
from simplecv.apis.view_exoego import compute_vertex_normals_batch
from simplecv.camera_parameters import PinholeParameters
from simplecv.data.skeleton.coco_133 import (
    COCO_133_ID2NAME,
    COCO_133_IDS,
    COCO_133_LINKS,
    LEFT_HAND_IDX,
    RIGHT_HAND_IDX,
)
from simplecv.data.skeleton.mediapipe import MEDIAPIPE_ID2NAME, MEDIAPIPE_IDS, MEDIAPIPE_LINKS
from simplecv.ops.mano.mano_np import MANOLayerNP
from simplecv.ops.mano.optim_jax_single_shape import (
    OptimShapeInput,
    OptimShapeResult,
    PoseShapeOptimConfig,
    SingleHandShapeOptim,
)
from simplecv.ops.triangulate import batch_triangulate
from simplecv.rerun_log_utils import (
    Points2DWithConfidence,
    Points3DWithConfidence,
    RerunTyroConfig,
    confidence_scores_to_rgb,
    log_pinhole,
    log_video,
)
from simplecv.video_io import MultiVideoReader
from wilor_nano.hand_detection import DetectionResult, HandDetector, HandDetectorConfig
from wilor_nano.hand_keypoints import HandKeypointDetectorConfig, KeypointResults, WilorHandKeypointDetector

from annotation_example.api.calibrate_mv_videos import MultiViewCalibrator, MVCalibResults
from annotation_example.rr_blueprints import create_view_container

np.set_printoptions(suppress=True)

SUPPORTED_IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg")
device = "cuda" if torch.cuda.is_available() else "cpu"


def timestamp_to_frame_index(
    frame_timestamps_ns: Int[ndarray, "num_frames"],
    target_timestamp_ns: int,
) -> int:
    """Map a nanosecond timestamp to the frame index at or immediately before it.

    The frame timestamps originate from `rr.AssetVideo.read_frame_timestamps_nanos`, which
    derives them from the video container timing. Those values are rarely exact multiples
    of 1/FPS, so we consistently **floor** to the last frame that does not exceed the
    requested timestamp.
    """
    frame_times: Int[ndarray, "num_frames"] = frame_timestamps_ns
    if frame_times.size == 0:
        msg = "Cannot convert timestamp to frame index for an empty timestamp array"
        raise ValueError(msg)

    # Clamp to valid range before applying the rounding strategy.
    if target_timestamp_ns <= int(frame_times[0]):
        return 0

    last_idx: int = int(frame_times.shape[0] - 1)
    if target_timestamp_ns >= int(frame_times[last_idx]):
        return last_idx

    insert_pos: int = int(np.searchsorted(frame_times, target_timestamp_ns, side="right")) - 1
    if insert_pos < 0:
        return 0
    return insert_pos


def frame_index_to_timestamp(frame_timestamps_ns: Int[ndarray, "num_frames"], frame_index: int) -> int:
    """Return the nanosecond timestamp associated with a frame index."""
    if frame_index < 0 or frame_index >= int(frame_timestamps_ns.shape[0]):
        msg = f"frame_index {frame_index} is outside the valid range [0, {frame_timestamps_ns.shape[0] - 1}]"
        raise IndexError(msg)
    timestamp_ns: int = int(frame_timestamps_ns[frame_index])
    return timestamp_ns


def set_annotation_context(recording: rr.RecordingStream | None = None) -> None:
    rr.log(
        "/",
        rr.AnnotationContext(
            [
                rr.ClassDescription(
                    info=rr.AnnotationInfo(id=0, label="L", color=(0, 0, 255)),
                    keypoint_annotations=[
                        rr.AnnotationInfo(id=id, label=name) for id, name in MEDIAPIPE_ID2NAME.items()
                    ],
                    keypoint_connections=MEDIAPIPE_LINKS,
                ),
                rr.ClassDescription(
                    info=rr.AnnotationInfo(id=1, label="R", color=(255, 0, 0)),
                    keypoint_annotations=[
                        rr.AnnotationInfo(id=id, label=name) for id, name in MEDIAPIPE_ID2NAME.items()
                    ],
                    keypoint_connections=MEDIAPIPE_LINKS,
                ),
                rr.ClassDescription(
                    info=rr.AnnotationInfo(id=2, label="Coco Wholebody", color=(0, 0, 255)),
                    keypoint_annotations=[
                        rr.AnnotationInfo(id=id, label=name) for id, name in COCO_133_ID2NAME.items()
                    ],
                    keypoint_connections=COCO_133_LINKS,
                ),
            ]
        ),
        static=True,
        recording=recording,
    )


@dataclass
class HandCalibConfig:
    rr_config: RerunTyroConfig
    image_dir: Path | None = None
    """Directory containing input images."""
    videos_dir: Path | None = None
    """Directory containing input videos."""
    ts_nano: int | None = None
    """Optional absolute timestamp in nanoseconds for selecting a video frame (floored to the nearest prior frame)."""
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


class ParsedInputs(NamedTuple):
    """NamedTuple containing parsed input data for calibration.

    Attributes
    ----------
    rgb_list : list[UInt8[ndarray, "H W 3"]]
        List of RGB images, each a 3D numpy array with shape (H, W, 3) and dtype uint8.
    input_log_paths : list[Path]
        List of input log paths (image paths for images, video paths for videos).
    exo_ts : Int[ndarray, "num_frames"] | None
        Timestamp array for the shortest video (in nanoseconds), or None for images.
    """

    rgb_list: list[UInt8[ndarray, "H W 3"]]
    input_log_paths: list[Path]
    exo_ts: Int[ndarray, "num_frames"] | None


def parse_input(
    input_type: Literal["videos", "images"], config: HandCalibConfig, parent_log_path: Path, timeline: str
) -> ParsedInputs:
    """
    Parses input data based on the specified type, either images or videos, and returns a ParsedInputs NamedTuple.

    This function handles two input types:
    - For "images": Scans the configured image directory for supported image file extensions,
      sorts them naturally, loads them as BGR images using OpenCV, converts to RGB, and sets input log paths for images.
    - For "videos": Finds video files in the configured videos directory, logs each video's frame timestamps,
      selects frames either via explicit index or via timestamp conversion (flooring to the closest prior frame),
      converts BGR frames to RGB, and sets input log paths for videos.

    Parameters
    ----------
    input_type : Literal["videos", "images"]
        The type of input to parse. Must be either "videos" or "images".
    config : HandCalibConfig
        Configuration object containing paths and settings, such as image_dir, videos_dir, ts_idx, and ts_ns.
    parent_log_path : Path
        The parent directory path for logging input data.
    timeline : str
        The timeline identifier for logging video frames (used only for videos).

    Returns
    -------
    ParsedInputs
        A NamedTuple containing:
        - rgb_list: List of RGB images, each a 3D numpy array with shape (H, W, 3) and dtype uint8.
        - input_log_paths: List of input log paths (image paths for images, video paths for videos).
        - exo_ts: Timestamp array for the shortest video (in nanoseconds), or None for images.
        - selected_timestamp_ns: Timestamp (nanoseconds) for the selected frame, or None for images.

    Raises
    ------
    AssertionError
        If no images are found in the image directory for "images" input type.
        If no videos are found in the videos directory for "videos" input type.

    Notes
    -----
    - Supported image extensions are defined in SUPPORTED_IMAGE_EXTENSIONS (e.g., .png, .jpg, .jpeg).
    - For videos, only .mp4 files are considered.
    - The function assumes OpenCV (cv2) and related utilities are available.
    - Timestamps are logged in nanoseconds for video frames.
    """
    exo_ts_list: list[Int[ndarray, "num_frames"]] | None = None
    mv_reader: MultiVideoReader | None = None
    bgr_list: list[UInt8[ndarray, "H W 3"]]
    match input_type:
        case "images":
            image_paths = []

            for ext in SUPPORTED_IMAGE_EXTENSIONS:
                image_paths.extend(config.image_dir.glob(f"*{ext}"))
            image_paths: list[Path] = natsorted(image_paths)
            assert len(image_paths) > 0, (
                f"No images found in {config.image_dir} in supported formats {SUPPORTED_IMAGE_EXTENSIONS}"
            )

            bgr_list: list[UInt8[ndarray, "H W 3"]] = [cv2.imread(str(image_path)) for image_path in image_paths]
            input_log_paths: list[Path] = [
                parent_log_path / "exo" / f"camera_{i}" / "pinhole" / "image" for i in range(len(bgr_list))
            ]
        case "videos":
            video_path_list: list[Path] = natsorted(config.videos_dir.glob("*.mp4"))
            input_log_paths: list[Path] = [
                parent_log_path / "exo" / f"camera_{i}" / "pinhole" / "video" for i in range(len(video_path_list))
            ]
            exo_ts_list: list[Int[ndarray, "num_frames"]] = []
            assert len(video_path_list) > 0, f"No videos found in {config.videos_dir}"
            for i, video_path in enumerate(video_path_list):
                exo_ts: Int[ndarray, "num_frames"] = log_video(
                    video_path=video_path,
                    video_log_path=input_log_paths[i],
                    timeline=timeline,
                )
                exo_ts_list.append(exo_ts)

            mv_reader = MultiVideoReader(video_path_list)

    min_exo_ts: Int[ndarray, "num_frames"] | None = min(exo_ts_list, key=len) if input_type == "videos" else None

    if input_type == "videos":
        assert mv_reader is not None, "MultiVideoReader must be initialized for video inputs"
        assert min_exo_ts is not None, "Timestamps are required for video inputs"
        if config.ts_nano is not None:
            ts_nanos: int = config.ts_nano
            frame_index: int = timestamp_to_frame_index(min_exo_ts, ts_nanos)
        else:
            frame_index = 0
        bgr_list = mv_reader[frame_index]

    rgb_list: list[UInt8[ndarray, "H W 3"]] = [cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB) for bgr in bgr_list]
    return ParsedInputs(rgb_list=rgb_list, input_log_paths=input_log_paths, exo_ts=min_exo_ts)


def main(config: HandCalibConfig) -> None:
    parent_log_path = Path("world")
    timeline = "video_time"

    if config.image_dir is None and config.videos_dir is None:
        raise ValueError("Either image or videos directory must be specified")

    ###################
    # 0. Parse inputs #
    ###################
    input_type: Literal["videos", "images"] = "images" if config.image_dir is not None else "videos"

    parsed_inputs: ParsedInputs = parse_input(input_type, config, parent_log_path, timeline)
    rgb_list: list[UInt8[ndarray, "H W 3"]] = parsed_inputs.rgb_list
    input_log_paths: list[Path] = parsed_inputs.input_log_paths
    exo_ts: Int[ndarray, "num_frames"] | None = parsed_inputs.exo_ts

    start: float = timer()

    ##########################
    # 1. Setup Rerun Logging #
    ##########################
    final_container: rrb.Container = create_view_container(
        parent_log_path=parent_log_path, num_images=len(rgb_list), show_videos=config.videos_dir is not None
    )
    blueprint: rrb.Blueprint = rrb.Blueprint(final_container, collapse_panels=True)
    rr.send_blueprint(blueprint=blueprint)
    set_annotation_context()
    rr.log(f"{parent_log_path}", rr.ViewCoordinates.RFU, static=True)
    rr.set_time(timeline, duration=0)

    ############################
    # 2. Calibrate Exo Cameras #
    ############################
    for idx, rgb in enumerate(rgb_list):
        rr.log(
            f"{parent_log_path}/exo/camera_{idx}/pinhole/image",
            rr.Image(rgb, color_model=rr.ColorModel.RGB),
            static=True,
        )
    mv_calibrator: MultiViewCalibrator = MultiViewCalibrator(refine_depth_maps=config.refine_depth_maps)
    results: MVCalibResults = mv_calibrator(rgb_list=rgb_list)

    pinhole_param_list: list[PinholeParameters] = results.pinhole_param_list
    pcd: o3d.geometry.PointCloud = results.pcd
    for pinhole, input_log_path in zip(pinhole_param_list, input_log_paths, strict=True):
        cam_log_path: Path = input_log_path.parent.parent
        log_pinhole(camera=pinhole, cam_log_path=cam_log_path, image_plane_distance=0.1, static=True)

    filtered_points: Float[ndarray, "final_points 3"] = np.asarray(pcd.points, dtype=np.float32)
    filtered_colors: Float[ndarray, "final_points 3"] = np.asarray(pcd.colors, dtype=np.float32)

    rr.log(
        f"{parent_log_path}/pointcloud",
        rr.Points3D(
            filtered_points,
            colors=filtered_colors,
        ),
        static=True,
    )

    ################################
    # 3. Calibrate Mano Parameters #
    ################################
    hand_detection_engine = HandDetector(HandDetectorConfig(verbose=False))
    hand_keypoint_engine = WilorHandKeypointDetector(HandKeypointDetectorConfig(verbose=False))

    uvc_coco_list: list[Float[ndarray, "n_frames n_kpts=133 3"]] = []
    Pall_exo: Float[ndarray, "num_frames 3 4"] = np.stack([pinhole.projection_matrix for pinhole in pinhole_param_list])
    for camera_idx, rgb_hw3 in enumerate(rgb_list):
        rr.set_time(timeline=timeline, duration=config.ts_nano * 1e-9 if config.ts_nano is not None else 0)
        det_result: DetectionResult = hand_detection_engine(rgb_hw3=rgb_hw3, hand_conf=0.3)
        left_xyxy: Float[ndarray, "1 4"] | None = det_result.left_xyxy
        right_xyxy: Float[ndarray, "1 4"] | None = det_result.right_xyxy

        pinhole_path = parent_log_path / "exo" / f"camera_{camera_idx}" / "pinhole"
        image_path = pinhole_path / "image"

        uvc_coco: Float[ndarray, "133 3"] = np.full((133, 3), np.nan, dtype=np.float32)
        for hand, xyxy in [("left", left_xyxy), ("right", right_xyxy)]:
            if xyxy is None:
                continue
            hand_path: Path = image_path / hand

            kpts_results: KeypointResults = hand_keypoint_engine(rgb_hw3=rgb_hw3, xyxy=xyxy, handedness=hand)
            uv: Float[ndarray, "n_frames=1 n_kpts=21 2"] = kpts_results.keypoints_2d
            conf: Float[ndarray, "n_frames=1 n_kpts=21"] = kpts_results.scores
            conf_values: Float[np.ndarray, "n_kpts"] = conf[0].astype(np.float32)
            mean_conf: float = float(np.nanmean(conf_values))
            uv_filtered: Float[np.ndarray, "n_kpts 2"] = uv[0].copy()
            # filter out low confidence points
            if mean_conf < 0.5:
                uv_filtered[:] = np.nan
                conf_values[:] = 0.0
            else:
                valid_mask: np.ndarray = conf_values >= 0.3
                uv_filtered[~valid_mask] = np.nan
                conf_values[~valid_mask] = 0.0

            conf_colors: UInt8[ndarray, "n_frames=1 n_kpts=21 3"] = confidence_scores_to_rgb(
                confidence_scores=conf_values[np.newaxis, :, np.newaxis]
            )

            fill_idx = RIGHT_HAND_IDX if hand == "right" else LEFT_HAND_IDX
            uvc_coco[fill_idx, :2] = uv_filtered
            uvc_coco[fill_idx, 2] = conf_values
            rr.log(
                f"{hand_path}/bbox",
                rr.Boxes2D(
                    array=xyxy,
                    array_format=rr.Box2DFormat.XYXY,
                    class_ids=0 if hand == "left" else 1,
                    show_labels=True,
                ),
            )
            rr.log(
                f"{hand_path}/keypoints",
                Points2DWithConfidence(
                    positions=uv_filtered,
                    confidences=conf_values,
                    class_ids=0 if hand == "left" else 1,
                    keypoint_ids=MEDIAPIPE_IDS,
                    show_labels=False,
                    colors=conf_colors[0],
                ),
            )
        uvc_coco_list.append(uvc_coco)
    # triangulate
    uvc_coco_batch: Float[ndarray, "n_views n_kpts=133 3"] = np.stack(uvc_coco_list)
    # can't be nan as it messes with triangulation, so convert everywhere thats a nan to zero
    uvc_triangulate_batch: Float[ndarray, "n_views n_kpts=133 3"] = np.nan_to_num(uvc_coco_batch.copy(), nan=0.0)
    xyzc: Float[ndarray, "n_kpts=133 4"] = batch_triangulate(uvc_triangulate_batch, Pall_exo, min_views=2)
    xyz: Float[ndarray, "n_kpts=133 3"] = xyzc[:, :3]
    conf_values: Float[ndarray, "n_kpts"] = xyzc[:, 3]
    conf_colors: UInt8[ndarray, "1 n_kpts 3"] = confidence_scores_to_rgb(conf_values[:, np.newaxis][np.newaxis, ...])

    rr.log(
        f"{parent_log_path}/wb_keypoints",
        Points3DWithConfidence(
            positions=xyz,
            confidences=conf_values,
            class_ids=2,
            keypoint_ids=COCO_133_IDS,
            show_labels=False,
            colors=conf_colors[0],
        ),
    )

    hand_side = "right"
    n_frames_optim: int = 1
    uv_exo_stack: Float[ndarray, "n_frames=1 n_views n_kpts=133 2"] = uvc_coco_batch[np.newaxis, :, :, 0:2]
    optim_shape_cfg = PoseShapeOptimConfig(
        Pall=Pall_exo, hand_side=hand_side, n_frames_optim=n_frames_optim, n_optim_iters=30
    )
    optimizer_shape = SingleHandShapeOptim(config=optim_shape_cfg)
    # take the first 30 frames and feed them into the optimizer
    optim_shape_input: OptimShapeInput = OptimShapeInput(
        uv_pred=uv_exo_stack[:n_frames_optim],
        beta_init=np.zeros((10,), dtype=np.float32),
        so3_init=np.zeros((n_frames_optim, 48), dtype=np.float32),
        trans_init=np.array([[0.0, 0.0, 0.6]]).repeat(n_frames_optim, axis=0).astype(np.float32),
    )
    optim_shape_tuple: tuple[OptimShapeResult, LevenbergMarquardtState] = optimizer_shape(optim_shape_input)
    optim_shape_result: OptimShapeResult = optim_shape_tuple[0]

    beta_optim: Float[ndarray, "10"] = optim_shape_result.beta_optim
    mano_layer = MANOLayerNP(side=hand_side, betas=beta_optim)

    mano_outputs: tuple[
        Float32[ndarray, "n_frames n_verts=778 3"],
        Float32[ndarray, "n_frames n_joints=21 3"],
    ] = mano_layer(optim_shape_result.so3_optim, optim_shape_result.trans_optim)
    verts: Float32[ndarray, "n_frames n_verts=778 3"] = mano_outputs[0]
    _: Float32[ndarray, "n_frames n_joints=21 3"] = mano_outputs[1]

    verts_np: Float32[ndarray, "n_frames n_verts=778 3"] = verts
    faces_np: Int[ndarray, "n_faces=1538 3"] = mano_layer.f.astype(np.int32)
    vertex_normals: Float32[ndarray, "n_frames n_verts=778 3"] = compute_vertex_normals_batch(verts_np[0:1], faces_np)

    rr.log(
        f"{parent_log_path}/{hand_side}_hand_mano",
        rr.Mesh3D(
            vertex_positions=verts_np[0],
            triangle_indices=faces_np,
            vertex_normals=vertex_normals[0],
            albedo_factor=(0, 0, 255, 255),
        ),
    )
    print(f"Inference completed in {timer() - start:.2f} seconds")
