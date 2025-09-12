# label_app/controller.py
import uuid
from dataclasses import replace
from pathlib import Path
from typing import Literal, assert_never

import gradio as gr
import numpy as np
import open3d as o3d
import rerun as rr
import rerun.blueprint as rrb
from einops import rearrange
from jaxtyping import Bool, Float32, Int, UInt8
from natsort import natsorted
from numpy import ndarray
from simplecv.rerun_log_utils import log_video
from simplecv.video_io import MultiVideoReader

from annotation_example.dbt_ui.engine import Engine
from annotation_example.dbt_ui.state import AppState, CurrentPrediction


def get_recording(
    recording_id: uuid.UUID,
    application_id: str = "Detection By Tracking Annotation",
    rrd_path: Path | None = None,
) -> rr.RecordingStream:
    recording = rr.RecordingStream(application_id=application_id, recording_id=recording_id)
    if rrd_path is not None:
        rr.set_sinks([rr.FileSink(path=rrd_path)], recording=recording)
    return recording


def create_dbt_blueprint(
    recording: rr.RecordingStream, state: AppState, active_tab: Literal["Info", "Annotations"] = "Info"
) -> rrb.Blueprint:
    # Create a DBT blueprint for the given recording and app state.
    info_tab = rrb.TextDocumentView(name="Info")
    # Use a container for the Annotations tab so we can arrange multiple views.
    # NOTE: Tabs.active_tab only matches by name for View children, not Containers.
    #       When the child is a Container, we must select the tab by index instead.

    annotation_tab = rrb.Spatial3DView()
    if state.rr_log_paths.video_log_paths is not None:
        spatial_2d_views: rrb.Horizontal = rrb.Horizontal(
            contents=[rrb.Spatial2DView(origin=video_log_path) for video_log_path in state.rr_log_paths.video_log_paths]
        )
        annotation_tab = rrb.Vertical(
            contents=[annotation_tab, spatial_2d_views], name="Annotations", row_shares=[3, 1]
        )

    # Map requested tab name to index to support container child. If using directly it will fail to match.
    match active_tab:
        case "Info":
            active_idx = 0
        case "Annotations":
            active_idx = 1
        case _:
            assert_never(active_tab)
    blueprint: rrb.Blueprint = rrb.Blueprint(rrb.Tabs(info_tab, annotation_tab, active_tab=active_idx))
    return blueprint


def _extract_zip_to_videos_dir(zipfile_path: Path) -> Path:
    import zipfile

    # Strategy:
    # - If the zip has a single top-level directory, extract into the parent dir
    #   to avoid double-nesting (e.g. .../lg-videos/lg-videos/...).
    # - Otherwise, extract into a dedicated directory named after the zip stem.
    with zipfile.ZipFile(zipfile_path, "r") as zip_ref:
        members: list[str] = [m for m in zip_ref.namelist() if m and m.strip("/")]
        # Determine top-level entries
        top_levels = {m.split("/", 1)[0] for m in members}
        single_top: bool = len(top_levels) == 1
        top_name: str | None = next(iter(top_levels)) if single_top else None
        # Consider it a directory if there are entries like "top_name/..."
        is_dir_like: bool = single_top and any(m.startswith(f"{top_name}/") for m in members if m != top_name)

        if single_top and is_dir_like and top_name is not None:
            # Extract beside the zip (into parent) to avoid creating .../<stem>/<stem>/...
            target_base: Path = zipfile_path.parent
            print(f"Extracting {zipfile_path} to {target_base} (preserving top-level folder '{top_name}')")
            zip_ref.extractall(target_base)
            videos_dir: Path = target_base / top_name
        else:
            # Mixed contents or a single file at top-level: extract into a dedicated dir
            extract_dir: Path = zipfile_path.parent / zipfile_path.stem
            print(f"Extracting {zipfile_path} to {extract_dir}")
            extract_dir.mkdir(parents=True, exist_ok=True)
            zip_ref.extractall(extract_dir)
            videos_dir = extract_dir

    assert videos_dir.exists(), f"Expected {videos_dir} to exist after extraction."
    print(f"Extracted to: {videos_dir}")
    return videos_dir


def _mv_calibrate(
    state: AppState,
    rgb_list: list[UInt8[ndarray, "H W 3"]],
    device: str = "cuda",
    refine_depth_maps: bool = True,
):
    parent_log_path: Path = state.rr_log_paths.parent_log_path
    recording: rr.RecordingStream = get_recording(state.recording_id, rrd_path=state.rrd_save_path)
    rr.log(f"{parent_log_path}", rr.ViewCoordinates.RFU, static=True, recording=recording)
    rr.set_time(state.rr_log_paths.timeline, duration=0, recording=recording)

    from monopriors.multiview_models.vggt_model import MultiviewPred, VGGTPredictor, robust_filter_confidences
    from monopriors.relative_depth_models import (
        RelativeDepthPrediction,
        get_relative_predictor,
    )
    from monopriors.relative_depth_models.base_relative_depth import BaseRelativePredictor

    from annotation_example.api.calibrate_mv_videos import (
        compute_scale_and_shift,
        depth_edges_mask,
        estimate_voxel_size,
        log_pinhole,
        mv_pred_to_pointcloud,
        orient_mv_pred_list,
    )

    vggt_predictor = VGGTPredictor(
        device=device,
        preprocessing_mode="pad",
    )
    mv_pred_list: list[MultiviewPred] = vggt_predictor(rgb_list=rgb_list)
    mv_pred_list = orient_mv_pred_list(mv_pred_list)

    pointcloud: Float32[ndarray, "num_points 3"] = mv_pred_to_pointcloud(mv_pred_list)
    rgb_stack: UInt8[ndarray, "num_points 3"] = np.concatenate(
        [rearrange(mv_pred.rgb_image, "h w c -> (h w) c") for mv_pred in mv_pred_list]
    )

    # create depth confidence values using robust filtering for top keep percentile
    depth_confidences: list[UInt8[ndarray, "H W"]] = [
        robust_filter_confidences(mv_pred.confidence_mask, keep_top_percent=0.3) for mv_pred in mv_pred_list
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
        recording=recording,
    )
    if refine_depth_maps:
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

        if refine_depth_maps:
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
            recording=recording,
        )

        rr.log(
            f"{pinhole_log_path}/image",
            rr.Image(mv_pred.rgb_image, color_model=rr.ColorModel.RGB).compress(),
            static=True,
            recording=recording,
        )
        rr.log(
            f"{pinhole_log_path}/confidence",
            rr.Image(depth_conf, color_model=rr.ColorModel.L).compress(),
            static=True,
            recording=recording,
        )
        rr.log(
            f"{pinhole_log_path}/filtered_depth",
            rr.DepthImage(filtered_depth_map, meter=1),
            static=True,
            recording=recording,
        )
        rr.log(
            f"{pinhole_log_path}/depth",
            rr.DepthImage(depth_map, meter=1),
            static=True,
            recording=recording,
        )
        if refine_depth_maps:
            rr.log(
                f"{pinhole_log_path}/refined_depth",
                rr.DepthImage(metric_depth, meter=1),
                static=True,
                recording=recording,
            )

    if refine_depth_maps:
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
            recording=recording,
        )


class Controller:
    """Glue between UI events, Engine calls, pure state transitions, and Rerun logging."""

    def __init__(
        self,
        engine: Engine,
    ):
        self.engine: Engine = engine
        # only show info on the start, so we'll switch to Annotations tab on video upload
        self.tab_name: Literal["Info", "Annotations"] = "Info"

    # ---- handlers wired by the panel ----
    def log_state(self, state):
        yield from self._log_state(state)

    def _log_state(self, state: gr.State | AppState):
        """Log the current image to Rerun."""
        recording: rr.RecordingStream = get_recording(state.recording_id)
        blueprint: rrb.Blueprint = create_dbt_blueprint(recording, state, active_tab=self.tab_name)
        rr.send_blueprint(blueprint, recording=recording)
        rr.log("info", rr.TextDocument("Press 'Next' to start annotating!"), static=True, recording=recording)

        # log the current prediction if it exists
        if state.current_prediction is not None:
            # set the timeline
            rr.set_time(
                timeline=state.rr_log_paths.timeline, duration=state.current_time_ns * 1e-9, recording=recording
            )
            print("Logging current prediction to Rerun")
            current_pred: CurrentPrediction = state.current_prediction
            if current_pred.detection_results.right_xyxy is not None:
                rr.log(
                    f"{state.rr_log_paths.parent_log_path}/video/right_xyxy",
                    rr.Boxes2D(array=current_pred.detection_results.right_xyxy, array_format=rr.Box2DFormat.XYXY),
                    recording=recording,
                )
            if current_pred.detection_results.left_xyxy is not None:
                rr.log(
                    f"{state.rr_log_paths.parent_log_path}/video/left_xyxy",
                    rr.Boxes2D(array=current_pred.detection_results.left_xyxy, array_format=rr.Box2DFormat.XYXY),
                    recording=recording,
                )

        yield recording.binary_stream().read(), state

    def initialize_rrd(self, video, state):
        yield from self._initialize_rrd(video, state)

    def _initialize_rrd(self, zip_path: str | None, state: gr.State | AppState):
        if zip_path is None:
            # create a new recording id to clear out any previous video state
            state: AppState = replace(state, recording_id=uuid.uuid4(), video_path=None)
            yield None, state
            return
        zip_path = Path(zip_path)
        videos_dir: Path = _extract_zip_to_videos_dir(zip_path)
        assert videos_dir.exists(), f"Video path {videos_dir} does not exist!"
        # make sure that we have videos in the directory
        video_path_list: list[Path] = natsorted(videos_dir.glob("*.mp4"))
        assert len(video_path_list) > 0, "No videos found in uploaded zip"
        recording: rr.RecordingStream = get_recording(state.recording_id)

        exo_timestamps: list[Int[ndarray, "num_frames"]] = []
        video_log_paths: list[Path] = []
        for i, video_path in enumerate(video_path_list):
            video_log_path = state.rr_log_paths.parent_log_path / f"camera_{i}" / "pinhole" / "video"
            frame_timestamps_ns: Int[ndarray, "num_frames"] = log_video(
                video_path=video_path,
                video_log_path=video_log_path,
                timeline=state.rr_log_paths.timeline,
                recording=recording,
            )
            video_log_paths.append(video_log_path)
            exo_timestamps.append(frame_timestamps_ns)

        mv_reader = MultiVideoReader(video_path_list)
        ts_idx: int = 0
        bgr_list: list[UInt8[ndarray, "H W 3"]] = mv_reader[ts_idx]
        rgb_list: list[UInt8[ndarray, "H W 3"]] = [bgr[..., ::-1] for bgr in bgr_list]
        _mv_calibrate(state, rgb_list, device="cuda", refine_depth_maps=True)

        # update rr_paths in state
        rr_log_paths = state.rr_log_paths
        rr_log_paths.video_log_paths = video_log_paths
        state = replace(state, rr_log_paths=rr_log_paths)

        # switch to Annotations tab on video upload
        self.tab_name = "Annotations"
        blueprint: rrb.Blueprint = create_dbt_blueprint(recording, state, active_tab=self.tab_name)
        rr.send_blueprint(blueprint, recording=recording)
        # frame_timestamps_ns: Int[ndarray, "num_frames"] = log_video(
        #     video_path=video_path,
        #     video_log_path=state.rr_log_paths.parent_log_path / "video",
        #     timeline=state.rr_log_paths.timeline,
        #     recording=recording,
        # )
        # # Update state immutably with video path & frame timestamps
        # new_state: AppState = replace(state, video_path=video_path, frame_timestamps_ns=frame_timestamps_ns)
        yield recording.binary_stream().read(), state

    # def on_nav(self, state: AppState, kind: Action) -> AppState:
    #     """Handle navigation events."""
    #     new_state: AppState = reduce(state, kind)
    #     return new_state
