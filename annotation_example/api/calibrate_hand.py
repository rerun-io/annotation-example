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
from jaxtyping import Float, Int, UInt8
from monopriors.apis.multiview_calibration import MultiViewCalibrator, MultiViewCalibratorConfig, MVCalibResults
from natsort import natsorted
from numpy import ndarray
from simplecv.camera_parameters import PinholeParameters
from simplecv.data.skeleton.coco_133 import (
    COCO_133_ID2NAME,
    COCO_133_LINKS,
)
from simplecv.data.skeleton.mediapipe import MEDIAPIPE_ID2NAME, MEDIAPIPE_LINKS
from simplecv.ops.pc_utils import estimate_voxel_size
from simplecv.rerun_log_utils import (
    RerunTyroConfig,
    log_pinhole,
    log_video,
)
from simplecv.video_io import MultiVideoReader
from tqdm import tqdm
from wilor_nano.hand_detection import HandDetector, HandDetectorConfig
from wilor_nano.hand_keypoints import HandKeypointDetectorConfig, WilorHandKeypointDetector

from annotation_example.api.benchmark_hand_pipeline import (
    mv_reader_to_rgb_ts_batch,
)
from annotation_example.api.hand_tracker import log_mano_outputs
from annotation_example.hand_calibrator import HandCalibrationResult, HandCalibrator, HandCalibratorConfig
from annotation_example.mv_hand_tracker import (
    MultiHandState,
    MultiViewHandTracker,
    MultiViewHandTrackerConfig,
)
from annotation_example.rr_blueprints import create_view_container

np.set_printoptions(suppress=True)

SUPPORTED_IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg")
device = "cuda" if torch.cuda.is_available() else "cpu"


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
    input_type: Literal["videos", "images"],
    image_dir: Path | None,
    videos_dir: Path | None,
    calib_ts_nano: int | None,
    parent_log_path: Path,
    timeline: str,
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
    image_dir : Path | None
        Directory containing input images when ``input_type`` is ``"images"``.
    videos_dir : Path | None
        Directory containing input videos when ``input_type`` is ``"videos"``.
    calib_ts_nano : int | None
        Optional nanosecond timestamp used to select calibration frames for videos.
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
    bgr_list: list[UInt8[ndarray, "H W 3"]] | None = None
    match input_type:
        case "images":
            if image_dir is None:
                raise ValueError("Image input requested but 'image_dir' is not provided")
            resolved_image_dir: Path = image_dir
            image_paths: list[Path] = []

            for ext in SUPPORTED_IMAGE_EXTENSIONS:
                image_paths.extend(resolved_image_dir.glob(f"*{ext}"))
            image_paths = natsorted(image_paths)
            assert len(image_paths) > 0, (
                f"No images found in {resolved_image_dir} in supported formats {SUPPORTED_IMAGE_EXTENSIONS}"
            )

            bgr_list_images: list[UInt8[ndarray, "H W 3"]] = []
            for image_path in image_paths:
                bgr_image = cv2.imread(str(image_path))
                if bgr_image is None:
                    msg = f"Failed to read image '{image_path}'"
                    raise FileNotFoundError(msg)
                bgr_array_np: UInt8[ndarray, "H W 3"] = np.asarray(bgr_image, dtype=np.uint8)
                bgr_array: UInt8[ndarray, "H W 3"] = bgr_array_np
                bgr_list_images.append(bgr_array)
            bgr_list = bgr_list_images
            input_log_paths: list[Path] = [
                parent_log_path / "exo" / f"camera_{i}" / "pinhole" / "image" for i in range(len(bgr_list))
            ]
        case "videos":
            if videos_dir is None:
                raise ValueError("Video input requested but 'videos_dir' is not provided")
            video_dir: Path = videos_dir
            video_path_list: list[Path] = natsorted(video_dir.glob("*.mp4"))
            input_log_paths: list[Path] = [
                parent_log_path / "exo" / f"camera_{i}" / "pinhole" / "video" for i in range(len(video_path_list))
            ]
            exo_ts_entries: list[Int[ndarray, "num_frames"]] = []
            assert len(video_path_list) > 0, f"No videos found in {video_dir}"
            for i, video_path in enumerate(video_path_list):
                exo_ts: Int[ndarray, "num_frames"] = log_video(
                    video_path=video_path,
                    video_log_path=input_log_paths[i],
                    timeline=timeline,
                )
                exo_ts_entries.append(exo_ts)

            mv_reader = MultiVideoReader(video_path_list)
            exo_ts_list = exo_ts_entries

    min_exo_ts: Int[ndarray, "num_frames"] | None
    if exo_ts_list is not None and len(exo_ts_list) > 0:
        min_exo_ts = min(exo_ts_list, key=lambda arr: int(arr.shape[0]))
    else:
        min_exo_ts = None

    if input_type == "videos":
        assert mv_reader is not None, "MultiVideoReader must be initialized for video inputs"
        assert min_exo_ts is not None, "Timestamps are required for video inputs"
        if calib_ts_nano is not None:
            ts_nanos: int = calib_ts_nano
            frame_index: int = timestamp_to_frame_index(time_ns=ts_nanos, frame_timestamps_ns=min_exo_ts)
        else:
            frame_index = 0
        bgr_frames: list[np.ndarray] = mv_reader[frame_index]
        bgr_frames_list: list[UInt8[ndarray, "H W 3"]] = []
        for frame in bgr_frames:
            bgr_frame_np: UInt8[ndarray, "H W 3"] = np.asarray(frame, dtype=np.uint8)
            bgr_frame: UInt8[ndarray, "H W 3"] = bgr_frame_np
            bgr_frames_list.append(bgr_frame)
        bgr_list = bgr_frames_list

    if bgr_list is None:
        raise RuntimeError("Failed to load any input frames")

    rgb_list_converted: list[UInt8[ndarray, "H W 3"]] = []
    for bgr in bgr_list:
        rgb_np: UInt8[ndarray, "H W 3"] = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        rgb: UInt8[ndarray, "H W 3"] = rgb_np
        rgb_list_converted.append(rgb)
    rgb_list: list[UInt8[ndarray, "H W 3"]] = rgb_list_converted
    return ParsedInputs(rgb_list=rgb_list, input_log_paths=input_log_paths, exo_ts=min_exo_ts)


@dataclass
class HandCalibConfig:
    rr_config: RerunTyroConfig
    image_dir: Path | None = None
    """Directory containing input images."""
    videos_dir: Path | None = None
    """Directory containing input videos."""
    calib_ts_nano: int | None = None
    """Optional nanosecond timestamp used to select calibration frames for cameras and MANO."""
    max_frames: int | None = None
    """Maximum number of frames to process. If None, all frames are processed."""
    tracking_start_ts_nano: int | None = None
    """Optional nanosecond timestamp to start MANO tracking from; defaults to ``0`` when omitted."""
    tracking_max_frames: int | None = None
    """Optional number of frames to run MANO tracking for; defaults to all remaining frames."""
    mv_hand_config: MultiViewHandTrackerConfig = field(default_factory=MultiViewHandTrackerConfig)
    """Parameters forwarded to the multi-view hand tracker."""
    calib_confg: MultiViewCalibratorConfig = field(default_factory=MultiViewCalibratorConfig)
    """Parameters forwarded to the multi-view calibrator."""
    do_hand_calib: bool = True
    """Whether to perform hand calibration."""
    calib_hand_side: Literal["left", "right"] = "right"
    """Hand side to optimize during MANO calibration; choose 'left' or 'right'."""
    stage: Literal["calib", "track", "all"] = "all"
    """Which pipeline stage to run: calibration only, tracking only, or all stages."""


def main(config: HandCalibConfig) -> None:
    parent_log_path = Path("world")
    timeline = "video_time"

    if config.image_dir is None and config.videos_dir is None:
        raise ValueError("Either image or videos directory must be specified")

    ###################
    # 0. Parse inputs #
    ###################
    input_type: Literal["videos", "images"] = "images" if config.image_dir is not None else "videos"

    parsed_inputs: ParsedInputs = parse_input(
        input_type=input_type,
        image_dir=config.image_dir,
        videos_dir=config.videos_dir,
        calib_ts_nano=config.calib_ts_nano,
        parent_log_path=parent_log_path,
        timeline=timeline,
    )
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
    if config.stage not in ("calib", "all"):
        print("Skipping camera calibration as per configuration")
        return
    for idx, rgb in enumerate(rgb_list):
        rr.log(
            f"{parent_log_path}/exo/camera_{idx}/pinhole/image",
            rr.Image(rgb, color_model=rr.ColorModel.RGB),
            static=True,
        )
    mv_calibrator: MultiViewCalibrator = MultiViewCalibrator(parent_log_path=parent_log_path, config=config.calib_confg)
    results: MVCalibResults = mv_calibrator(rgb_list=rgb_list)

    pinhole_param_list: list[PinholeParameters] = results.pinhole_param_list
    pcd: o3d.geometry.PointCloud = results.pcd
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

    ################################
    # 3. Calibrate Mano Parameters #
    ################################
    if config.stage not in ("track", "all"):
        print("Skipping hand calibration and tracking as per configuration")
        print(f"Inference completed in {timer() - start:.2f} seconds")
        return
    hand_detection_engine = HandDetector(HandDetectorConfig(verbose=False))
    hand_keypoint_engine = WilorHandKeypointDetector(HandKeypointDetectorConfig(verbose=False))

    if config.do_hand_calib:
        hand_calibrator = HandCalibrator(
            hand_detector=hand_detection_engine,
            hand_keypoint_detector=hand_keypoint_engine,
            config=HandCalibratorConfig(
                mano_optim_iters=30, ts_nano=config.calib_ts_nano, hand_side=config.calib_hand_side, verbose=False
            ),
            parent_log_path=parent_log_path,
        )

    #######################################
    # 4. Prepare Tracking Data Sources    #
    #######################################
    if config.videos_dir is not None and exo_ts is not None:
        video_path_list: list[Path] = natsorted(config.videos_dir.glob("*.mp4"))
        mv_reader = MultiVideoReader(video_path_list)

        target_ts_nano: int = (
            config.calib_ts_nano if config.calib_ts_nano is not None else frame_index_to_timestamp(exo_ts, 0)
        )
        frame_index: int = timestamp_to_frame_index(target_ts_nano, exo_ts)
        frame_timestamp_ns: int = frame_index_to_timestamp(exo_ts, frame_index)
        frame_timestamp_seconds: float = frame_timestamp_ns * 1e-9
        rr.set_time(timeline=timeline, duration=frame_timestamp_seconds)

        rgb_ts_batch: UInt8[ndarray, "n_frames n_views H W 3"] = mv_reader_to_rgb_ts_batch(
            mv_reader=mv_reader,
            num_frames=1,
            ts_nanos=target_ts_nano,
            frame_timestamps_ns=exo_ts,
        )
        if config.do_hand_calib:
            hand_calib_result: HandCalibrationResult = hand_calibrator(
                exo_cam_list=pinhole_param_list,
                rgb_ts_batch=rgb_ts_batch,
                recording=config.rr_config.rec_stream,
            )
            beta: Float[ndarray, "10"] = hand_calib_result.mano.betas

        else:
            beta: Float[ndarray, "10"] = np.zeros((10,), dtype=np.float32)

        tracker_config: MultiViewHandTrackerConfig = config.mv_hand_config
        mv_hand_tracker = MultiViewHandTracker(
            config=tracker_config,
            hand_detector=hand_detection_engine,
            hand_keypoint_detector=hand_keypoint_engine,
            betas=beta,
            pinhole_param_list=pinhole_param_list,
            parent_log_path=parent_log_path,
        )

        tracking_reader: MultiVideoReader = MultiVideoReader(video_path_list)
        hand_state: MultiHandState = MultiHandState()

        total_frames: int = int(exo_ts.shape[0])
        tracking_start_ts: int = (
            config.tracking_start_ts_nano
            if config.tracking_start_ts_nano is not None
            else 0
        )
        start_index: int = timestamp_to_frame_index(tracking_start_ts, exo_ts)
        max_frames: int | None = config.tracking_max_frames
        end_index: int = total_frames if max_frames is None else min(total_frames, start_index + max_frames)

        if end_index <= start_index:
            frame_range: range = range(0)
        else:
            frame_range = range(start_index, end_index)

        frame_iter: Iterable[int]
        if mv_hand_tracker.config.verbose:
            frame_iter = cast(Iterable[int], tqdm(frame_range, total=len(frame_range)))
        else:
            frame_iter = frame_range

        #####################################
        # 5. Run Tracking And Log Outputs   #
        #####################################
        for frame_idx in frame_iter:
            ts_nano: int = int(exo_ts[frame_idx])
            rr.set_time(timeline=timeline, duration=ts_nano * 1e-9)

            bgr_views: list[UInt8[ndarray, "H W 3"]] = tracking_reader[frame_idx]
            rgb_views: list[UInt8[ndarray, "H W 3"]] = [
                cv2.cvtColor(bgr_hw3, cv2.COLOR_BGR2RGB) for bgr_hw3 in bgr_views
            ]
            rgb_batch: UInt8[ndarray, "n_views H W 3"] = np.stack(rgb_views, axis=0)

            hand_state: MultiHandState = mv_hand_tracker(
                rgb_batch=rgb_batch,
                pinhole_param_list=pinhole_param_list,
                hand_state=hand_state,
                recording=config.rr_config.rec_stream,
            )

            log_mano_outputs(
                hand_state=hand_state,
                tracker=mv_hand_tracker,
                pinhole_param_list=pinhole_param_list,
                parent_log_path=parent_log_path,
                recording=config.rr_config.rec_stream,
            )

    print(f"Inference completed in {timer() - start:.2f} seconds")
