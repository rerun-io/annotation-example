from dataclasses import replace
from pathlib import Path
from typing import NamedTuple

import cv2
import numpy as np
import open3d as o3d
import rerun as rr
from einops import rearrange
from jaxtyping import Bool, Float32, UInt8
from monopriors.multiview_models.vggt_model import MultiviewPred, VGGTPredictor, robust_filter_confidences
from monopriors.relative_depth_models import (
    RelativeDepthPrediction,
    get_relative_predictor,
)
from monopriors.relative_depth_models.base_relative_depth import BaseRelativePredictor
from numpy import ndarray
from simplecv.camera_parameters import PinholeParameters
from simplecv.data.skeleton.mediapipe import MEDIAPIPE_IDS
from simplecv.video_io import MultiVideoReader
from tqdm import tqdm
from wilor_nano.hand_detection import DetectionResult, HandDetector, HandDetectorConfig
from wilor_nano.hand_keypoints import FinalWilorPred, HandKeypointDetectorConfig, WilorHandKeypointDetector

from annotation_example.api.calibrate_mv_videos import (
    compute_scale_and_shift,
    depth_edges_mask,
    estimate_voxel_size,
    mv_pred_to_pointcloud,
    orient_mv_pred_list,
    segment_people,
)
from annotation_example.dbt_ui.recording_utils import get_recording
from annotation_example.dbt_ui.state import AppState, CurrentPrediction

try:
    from sam2.sam2_image_predictor import SAM2ImagePredictor
except ImportError:
    SAM2ImagePredictor = None  # type: ignore

try:
    from rtmlib import YOLOX
except ImportError:
    YOLOX = None  # type: ignore


class MVCalibResults(NamedTuple):
    pinhole_param_list: list[PinholeParameters]
    pcd: o3d.geometry.PointCloud


class MultiViewCalibrator:
    def __init__(self) -> None:
        self.device: str = "cuda"
        self.det_model = YOLOX(
            "https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/yolox_m_8xb8-300e_humanart-c2c7a14a.zip",
            model_input_size=(640, 640),
            backend="onnxruntime",
            device=self.device,
        )

        self.sam2_predictor = SAM2ImagePredictor.from_pretrained("facebook/sam2-hiera-large")
        self.vggt_predictor = VGGTPredictor(
            device=self.device,
            preprocessing_mode="pad",
        )

        self.refine_depth_maps: bool = True
        if self.refine_depth_maps:
            self.moge_predictor: BaseRelativePredictor = get_relative_predictor("MogeV1Predictor")(device="cuda")

    def __call__(
        self,
        state: AppState,
        rgb_list: list[UInt8[ndarray, "H W 3"]],
    ) -> MVCalibResults:
        parent_log_path: Path = state.rr_log_paths.parent_log_path

        mv_pred_list: list[MultiviewPred] = self.vggt_predictor(rgb_list)
        mv_pred_list: list[MultiviewPred] = orient_mv_pred_list(mv_pred_list)

        # Compute person segmentation masks per view. Keep for potential UI/analysis,
        # but do not alter confidences/depth with it here.
        segmask_list: list[Bool[np.ndarray, "H W"] | None] = []
        for rgb in rgb_list:
            bgr: UInt8[ndarray, "H W 3"] = rgb[..., ::-1]
            people_masks: Bool[ndarray, "H W"] | None = segment_people(
                bgr, det_model=self.det_model, sam_2_predictor=self.sam2_predictor, dilation=50
            )
            segmask_list.append(people_masks)

        pointcloud: Float32[ndarray, "num_points 3"] = mv_pred_to_pointcloud(mv_pred_list)
        rgb_stack: UInt8[ndarray, "num_points 3"] = np.concatenate(
            [rearrange(mv_pred.rgb_image, "h w c -> (h w) c") for mv_pred in mv_pred_list]
        )

        # create depth confidence values using robust filtering for top keep percentile
        depth_confidences: list[UInt8[ndarray, "H W"]] = [
            robust_filter_confidences(mv_pred.confidence_mask, keep_top_percent=30) for mv_pred in mv_pred_list
        ]

        # update depth_confidences to exclude people, create a totally new list so it doesn't modify the original
        new_depth_confidences = []
        for depth_conf, segmask in zip(depth_confidences, segmask_list, strict=True):
            if segmask is not None:
                new_depth_confidences.append(depth_conf * ~segmask)
            else:
                new_depth_confidences.append(depth_conf)

        depth_confidences = new_depth_confidences
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

        print(f"parent_log_path: {parent_log_path}")
        if self.refine_depth_maps:
            refined_depths_list: list[Float32[ndarray, "H W"]] = []

        mv_pred: MultiviewPred
        for mv_pred in mv_pred_list:
            depth_map: Float32[ndarray, "H W"] = mv_pred.depth_map
            depth_conf: UInt8[ndarray, "H W"] = depth_confidences[mv_pred_list.index(mv_pred)]
            # Filter depth
            filtered_depth_map: Float32[ndarray, "H W"] = np.where(depth_conf > 0, depth_map, 0)

            if self.refine_depth_maps:
                relative_pred: RelativeDepthPrediction = self.moge_predictor.__call__(
                    rgb=mv_pred.rgb_image, K_33=mv_pred.pinhole_param.intrinsics.k_matrix
                )

                scale, shift = compute_scale_and_shift(
                    relative_pred.depth, filtered_depth_map, mask=depth_conf > 0, scale_only=False
                )
                metric_depth: Float32[np.ndarray, "h w"] = relative_pred.depth.copy() * scale + shift
                # filter depth
                edges_mask: Bool[np.ndarray, "h w"] = depth_edges_mask(metric_depth, threshold=0.01)
                metric_depth: Float32[np.ndarray, "h w"] = metric_depth * ~edges_mask
                metric_depth = np.where(depth_conf > 0, metric_depth, 0)
                # remove people from metric depth
                if segmask_list[mv_pred_list.index(mv_pred)] is not None:
                    metric_depth: Float32[np.ndarray, "h w"] = metric_depth * ~segmask_list[mv_pred_list.index(mv_pred)]

                refined_depths_list.append(metric_depth)

        if self.refine_depth_maps:
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

        mv_calib_results: MVCalibResults = MVCalibResults(
            pinhole_param_list=[mv_pred.pinhole_param for mv_pred in mv_pred_list],
            pcd=pcd_ds,
        )
        return mv_calib_results


class Engine:
    """Used to hold neural network engines."""

    def __init__(
        self,
    ):
        self.hand_detection_engine = HandDetector(HandDetectorConfig(verbose=False))
        self.hand_keypoint_engine = WilorHandKeypointDetector(HandKeypointDetectorConfig(verbose=False))
        self._mv_reader: MultiVideoReader | None = None
        self.mv_calibrator = MultiViewCalibrator()

    @property
    def mv_reader(self) -> MultiVideoReader | None:
        return self._mv_reader

    @mv_reader.setter
    def mv_reader(self, value: MultiVideoReader | None) -> None:
        self._mv_reader = value

    def calibrate_mv(self, state: AppState, rgb_list: list[UInt8[ndarray, "H W 3"]]) -> MVCalibResults:
        return self.mv_calibrator(state, rgb_list)

    def predict_xyxy(self, state: AppState) -> AppState:
        # Convert current_time_ns to frame index using the logged frame timestamps.
        if state.frame_timestamps_ns is None or state.video_paths_list is None:
            raise RuntimeError("Video not loaded or frame timestamps missing in state.")

        frame_idx: int = time_to_frame_idx(state.current_time_ns, state.frame_timestamps_ns)

        # Read frame from the underlying video using MultiVideoReader (BGR)
        if self.mv_reader is None or self.mv_reader.video_paths != [state.video_paths_list]:
            self.mv_reader = MultiVideoReader([state.video_paths_list])
        bgr_frame = self.mv_reader[frame_idx][0]
        rgb_hw3: UInt8[ndarray, "h w 3"] = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)

        # Run your model on the current frame (example placeholder)
        det_result: DetectionResult = self.hand_detection_engine(rgb_hw3=rgb_hw3, hand_conf=0.3)
        # update the app state with the new prediction
        current_prediction: CurrentPrediction = (
            state.current_prediction if state.current_prediction is not None else CurrentPrediction()
        )
        current_prediction.detection_results = det_result

        state: AppState = replace(state, current_prediction=current_prediction)
        return state

    def predict_mv_xyxy(self, state):
        yield from self._predict_mv_xyxy(state)

    def _predict_mv_xyxy(self, state: AppState):
        recording: rr.RecordingStream = get_recording(state.recording_id)
        for ts_idx, ts in tqdm(enumerate(state.frame_timestamps_ns)):
            rr.set_time(state.rr_log_paths.timeline, duration=ts * 1e-9, recording=recording)
            bgr_list = self.mv_reader[ts_idx]
            for bgr, video_log_path in zip(bgr_list, state.rr_log_paths.video_log_paths, strict=True):
                rgb_hw3: UInt8[ndarray, "h w 3"] = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

                # Run your model on the current frame (example placeholder)
                det_result: DetectionResult = self.hand_detection_engine(rgb_hw3=rgb_hw3, hand_conf=0.3)

                if det_result.right_xyxy is not None:
                    right_xyxy = det_result.right_xyxy
                    kpts_results: FinalWilorPred = self.hand_keypoint_engine(
                        rgb_hw3=rgb_hw3, xyxy=right_xyxy, handedness="right"
                    )
                    rr.log(
                        f"{video_log_path}/right_xyxy",
                        rr.Boxes2D(array=right_xyxy, array_format=rr.Box2DFormat.XYXY),
                        recording=recording,
                    )
                    rr.log(
                        f"{video_log_path}/right_keypoints",
                        rr.Points2D(
                            positions=kpts_results.pred_keypoints_2d[0],
                            class_ids=0,
                            keypoint_ids=MEDIAPIPE_IDS,
                            show_labels=False,
                            colors=(0, 255, 0),
                        ),
                        recording=recording,
                    )
                else:
                    rr.log(
                        f"{video_log_path}/right_xyxy",
                        rr.Clear(recursive=True),
                        recording=recording,
                    )
                    rr.log(
                        f"{video_log_path}/right_keypoints",
                        rr.Clear(recursive=True),
                        recording=recording,
                    )

                if det_result.left_xyxy is not None:
                    left_xyxy = det_result.left_xyxy
                    kpts_results: FinalWilorPred = self.hand_keypoint_engine(
                        rgb_hw3=rgb_hw3, xyxy=left_xyxy, handedness="left"
                    )
                    rr.log(
                        f"{video_log_path}/left_xyxy",
                        rr.Boxes2D(array=left_xyxy, array_format=rr.Box2DFormat.XYXY),
                        recording=recording,
                    )
                    rr.log(
                        f"{video_log_path}/left_keypoints",
                        rr.Points2D(
                            positions=kpts_results.pred_keypoints_2d[0],
                            class_ids=0,
                            keypoint_ids=MEDIAPIPE_IDS,
                            show_labels=False,
                            colors=(0, 255, 0),
                        ),
                        recording=recording,
                    )
                else:
                    rr.log(
                        f"{video_log_path}/left_xyxy",
                        rr.Clear(recursive=True),
                        recording=recording,
                    )
        yield recording.binary_stream().read(), state


def time_to_frame_idx(time_ns: int, frame_timestamps_ns: np.ndarray) -> int:
    """Map a timestamp (ns) to the closest frame idx at-or-before that time.

    The mapping mirrors how VideoFrameReference columns are generated from
    AssetVideo.read_frame_timestamps_nanos().
    """

    idx: int = int(np.searchsorted(frame_timestamps_ns, time_ns, side="right") - 1)
    return max(0, min(idx, len(frame_timestamps_ns) - 1))
