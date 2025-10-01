from dataclasses import dataclass, field
from pathlib import Path
from timeit import default_timer as timer
from typing import Literal

import cv2
import numpy as np
import rerun as rr
import rerun.blueprint as rrb
import torch
from jaxtyping import Float32, Int, UInt8
from monopriors.apis.multiview_calibration import (
    MultiViewCalibrator,
    MultiViewCalibratorConfig,
    MVCalibResults,
    create_final_view,
)
from numpy import ndarray
from simplecv.ops.pc_utils import estimate_voxel_size
from simplecv.rerun_log_utils import RerunTyroConfig, log_video
from simplecv.video_io import MultiVideoReader

np.set_printoptions(suppress=True)

SUPPORTED_IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg")
device = "cuda" if torch.cuda.is_available() else "cpu"


@dataclass
class VGGTInferenceConfig:
    """Runtime options for VGGT-based multi-view inference and calibration."""

    rr_config: RerunTyroConfig
    """Rerun logging configuration."""
    image_dir: Path | None = None
    """Directory containing input images."""
    videos_dir: Path | None = None
    """Directory containing input videos."""
    ts_idx: int = 0
    """Timestep for video chosen frames."""
    mv_calibrator_config: MultiViewCalibratorConfig = field(default_factory=MultiViewCalibratorConfig)
    """Base calibrator configuration; `refine_depth_maps` overrides its refinement flag."""


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

    #####################################
    # 1. Setup Rerun related components #
    #####################################
    rgb_list: list[UInt8[ndarray, "H W 3"]] = [cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB) for bgr in bgr_list]
    start: float = timer()
    final_view: rrb.ContainerLike = create_final_view(
        parent_log_path=parent_log_path, num_images=len(rgb_list), show_videos=config.videos_dir is not None
    )
    blueprint = rrb.Blueprint(final_view, collapse_panels=True)
    rr.send_blueprint(blueprint=blueprint)
    rr.log(f"{parent_log_path}", rr.ViewCoordinates.RFU, static=True)
    rr.set_time(timeline, duration=0)

    ##############################
    # 2. Run MultiViewCalibrator #
    ##############################
    mv_calibrator = MultiViewCalibrator(parent_log_path, config=config.mv_calibrator_config)
    output: MVCalibResults = mv_calibrator(rgb_list=rgb_list, recording=None)

    ###################################################
    # 3. Log Final Output (Not Verbose always logged) #
    ###################################################
    pcd = output.pcd

    # Automatically determine optimal voxel size based on point cloud characteristics
    voxel_size: float = estimate_voxel_size(np.asarray(pcd.points, dtype=np.float32), target_points=500_000)
    pcd_ds = pcd.voxel_down_sample(voxel_size)

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

    print(f"Inference completed in {timer() - start:.2f} seconds")
