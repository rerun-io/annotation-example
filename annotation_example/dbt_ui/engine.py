from dataclasses import replace
from pathlib import Path
from typing import Literal

import cv2
import numpy as np
import rerun as rr
from jaxtyping import Bool, Float, Int, UInt8
from monopriors.apis.multiview_calibration import (
    MultiViewCalibrator,
    MultiViewCalibratorConfig,
    MVCalibResults,
)
from numpy import ndarray
from simplecv.camera_parameters import PinholeParameters
from simplecv.data.skeleton.mediapipe import MEDIAPIPE_IDS
from simplecv.ops.triangulate import batch_triangulate
from simplecv.rerun_log_utils import (
    Points2DWithConfidence,
    Points3DWithConfidence,
    confidence_scores_to_rgb,
)
from simplecv.video_io import MultiVideoReader
from tqdm import tqdm
from wilor_nano.hand_detection import DetectionResult, HandDetector, HandDetectorConfig
from wilor_nano.hand_keypoints import (
    HandKeypointDetectorConfig,
    KeypointResults,
    RTMPoseHandKeypointDetector,
    WilorHandKeypointDetector,
)

from annotation_example.dbt_ui.recording_utils import get_recording
from annotation_example.dbt_ui.state import AppState, CurrentPrediction

HAND_CONFIDENCE: float = 0.3
Handedness = Literal["left", "right"]
HAND_CLASS_IDS: dict[Handedness, int] = {"left": 0, "right": 1}
KEYPOINT_CONFIDENCE_THRESHOLD: float = 0.25


class Engine:
    """Used to hold neural network engines."""

    def __init__(
        self,
    ) -> None:
        self.hand_detection_engine = HandDetector(HandDetectorConfig(verbose=False))
        kpt_network: Literal["wilor", "rtmpose"] = "wilor"
        if kpt_network == "wilor":
            self.hand_keypoint_engine = WilorHandKeypointDetector(HandKeypointDetectorConfig(verbose=False))
        elif kpt_network == "rtmpose":
            self.hand_keypoint_engine = RTMPoseHandKeypointDetector(HandKeypointDetectorConfig(verbose=False))
        self.mv_calibrator: MultiViewCalibrator = MultiViewCalibrator(
            parent_log_path=Path("world"), config=MultiViewCalibratorConfig()
        )
        self._ego_mv_reader: MultiVideoReader | None = None
        self._exo_mv_reader: MultiVideoReader | None = None

    @property
    def exo_mv_reader(self) -> MultiVideoReader | None:
        return self._exo_mv_reader

    @exo_mv_reader.setter
    def exo_mv_reader(self, value: MultiVideoReader | None) -> None:
        self._exo_mv_reader = value

    @property
    def ego_mv_reader(self) -> MultiVideoReader | None:
        return self._ego_mv_reader

    @ego_mv_reader.setter
    def ego_mv_reader(self, value: MultiVideoReader | None) -> None:
        self._ego_mv_reader = value

    def _log_hand_prediction(
        self,
        *,
        recording: rr.RecordingStream,
        pinhole_log_path: Path,
        hand: Handedness,
        rgb_hw3: UInt8[ndarray, "h w 3"],
        xyxy: Float[ndarray, "1 4"] | None,
    ) -> Float[np.ndarray, "n_kpts 3"] | None:
        hand_path: Path = pinhole_log_path / hand
        if xyxy is None:
            rr.log(f"{hand_path}_xyxy", rr.Clear(recursive=True), recording=recording)
            rr.log(f"{hand_path}_keypoints", rr.Clear(recursive=True), recording=recording)
            return None

        kpts_results: KeypointResults = self.hand_keypoint_engine(rgb_hw3=rgb_hw3, xyxy=xyxy, handedness=hand)
        uv: Float[ndarray, "n_frames=1 n_kpts=21 2"] = kpts_results.keypoints_2d
        conf: Float[ndarray, "n_frames=1 n_kpts=21"] = kpts_results.scores
        conf_colors: UInt8[ndarray, "n_frames=1 n_kpts=21 3"] = confidence_scores_to_rgb(
            confidence_scores=conf[..., np.newaxis]
        )
        class_id: int = HAND_CLASS_IDS[hand]

        rr.log(
            f"{hand_path}_xyxy",
            rr.Boxes2D(
                array=xyxy,
                array_format=rr.Box2DFormat.XYXY,
                class_ids=class_id,
                show_labels=True,
            ),
            recording=recording,
        )
        uv_frame: Float[np.ndarray, "n_kpts 2"] = uv[0]
        conf_values: Float[np.ndarray, "n_kpts"] = conf[0].astype(np.float32)
        rr.log(
            f"{hand_path}_keypoints",
            Points2DWithConfidence(
                positions=uv_frame,
                confidences=conf_values,
                class_ids=class_id,
                keypoint_ids=MEDIAPIPE_IDS,
                show_labels=False,
                colors=conf_colors[0],
            ),
            recording=recording,
        )
        conf_thresholded: Float[np.ndarray, "n_kpts"] = np.where(
            conf_values >= KEYPOINT_CONFIDENCE_THRESHOLD,
            conf_values,
            0.0,
        ).astype(np.float32)
        uv_conf: Float[np.ndarray, "n_kpts 3"] = np.concatenate(
            (uv_frame, conf_thresholded[..., np.newaxis]),
            axis=-1,
        ).astype(np.float32)
        return uv_conf

    def _process_video_reader(
        self,
        *,
        bgr_frames: list[UInt8[ndarray, "h w 3"]],
        video_log_paths: list[Path],
        recording: rr.RecordingStream,
        collect_keypoints: bool = False,
    ) -> dict[Path, dict[Handedness, Float[np.ndarray, "n_kpts 3"]]]:
        collected: dict[Path, dict[Handedness, Float[np.ndarray, "n_kpts 3"]]] = {}
        for bgr, video_log_path in zip(bgr_frames, video_log_paths, strict=True):
            rgb_hw3: UInt8[ndarray, "h w 3"] = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            det_result: DetectionResult = self.hand_detection_engine(rgb_hw3=rgb_hw3, hand_conf=HAND_CONFIDENCE)
            pinhole_log_path: Path = video_log_path.parent
            right_uv_conf: Float[np.ndarray, "n_kpts 3"] | None = self._log_hand_prediction(
                recording=recording,
                pinhole_log_path=pinhole_log_path,
                hand="right",
                rgb_hw3=rgb_hw3,
                xyxy=det_result.right_xyxy,
            )
            left_uv_conf: Float[np.ndarray, "n_kpts 3"] | None = self._log_hand_prediction(
                recording=recording,
                pinhole_log_path=pinhole_log_path,
                hand="left",
                rgb_hw3=rgb_hw3,
                xyxy=det_result.left_xyxy,
            )
            if collect_keypoints:
                hand_data: dict[Handedness, Float[np.ndarray, "n_kpts 3"]] = {}
                if right_uv_conf is not None:
                    hand_data["right"] = right_uv_conf
                if left_uv_conf is not None:
                    hand_data["left"] = left_uv_conf
                if hand_data:
                    collected[video_log_path] = hand_data
        return collected

    def _triangulate_exo_views(
        self,
        *,
        hand_keypoints: dict[Path, dict[Handedness, Float[np.ndarray, "n_kpts 3"]]],
        pinhole_params_list: list[PinholeParameters],
        video_log_paths: list[Path],
        recording: rr.RecordingStream,
        parent_log_path: Path,
    ) -> None:
        triangulation_root: Path = parent_log_path / "triangulated"
        hand_order: tuple[Handedness, ...] = ("left", "right")

        for hand in hand_order:
            per_view_keypoints: list[Float[np.ndarray, "n_kpts 3"]] = []
            projection_matrices: list[Float[np.ndarray, "3 4"]] = []

            for pinhole_param, video_log_path in zip(pinhole_params_list, video_log_paths, strict=False):
                hand_data: dict[Handedness, Float[np.ndarray, "n_kpts 3"]] | None = hand_keypoints.get(video_log_path)
                if hand_data is None:
                    continue
                uv_conf: Float[np.ndarray, "n_kpts 3"] | None = hand_data.get(hand)
                if uv_conf is None:
                    continue

                per_view_keypoints.append(uv_conf.astype(np.float32))
                projection_matrices.append(pinhole_param.projection_matrix.astype(np.float32))

            if len(per_view_keypoints) < 2:
                rr.log(
                    str(triangulation_root / f"{hand}_hand"),
                    rr.Clear(recursive=True),
                    recording=recording,
                )
                continue

            keypoints_stack: Float[np.ndarray, "n_views n_kpts 3"] = np.stack(per_view_keypoints, axis=0)
            proj_stack: Float[np.ndarray, "n_views 3 4"] = np.stack(projection_matrices, axis=0)

            xyzc: Float[np.ndarray, "n_kpts 4"] = batch_triangulate(
                keypoints_2d=keypoints_stack,
                projection_matrices=proj_stack,
                min_views=2,
            ).astype(np.float32)

            confidence: Float[np.ndarray, "n_kpts"] = xyzc[:, 3]
            valid_mask: Bool[np.ndarray, "n_kpts"] = confidence > 0.0
            if not np.any(valid_mask):
                rr.log(
                    str(triangulation_root / f"{hand}_hand"),
                    rr.Clear(recursive=True),
                    recording=recording,
                )
                continue

            positions: Float[np.ndarray, "n_kpts 3"] = xyzc[:, :3].astype(np.float32)
            positions[~valid_mask] = np.nan
            confidence_for_colors: Float[np.ndarray, "1 n_kpts 1"] = confidence.astype(np.float32)[
                np.newaxis, :, np.newaxis
            ]
            colors_tensor: UInt8[np.ndarray, "1 n_kpts 3"] = confidence_scores_to_rgb(
                confidence_scores=confidence_for_colors
            )
            colors: UInt8[np.ndarray, "n_kpts 3"] = colors_tensor[0]
            confidence_values: Float[np.ndarray, "n_kpts"] = confidence.astype(np.float32)
            rr.log(
                str(triangulation_root / f"{hand}_hand"),
                Points3DWithConfidence(
                    positions=positions,
                    confidences=confidence_values,
                    colors=colors,
                    radii=0.005,
                    keypoint_ids=MEDIAPIPE_IDS,
                    class_ids=HAND_CLASS_IDS[hand],
                    show_labels=False,
                ),
                recording=recording,
            )

    def predict_xyxy(self, state: AppState) -> AppState:
        # Convert current_time_ns to frame index using the logged frame timestamps.
        if state.frame_timestamps_ns is None or state.video_paths_list is None:
            raise RuntimeError("Video not loaded or frame timestamps missing in state.")

        frame_idx: int = time_to_frame_idx(state.current_time_ns, state.frame_timestamps_ns)

        # Read frame from the underlying video using MultiVideoReader (BGR)
        if self.ego_mv_reader is None or self.ego_mv_reader.video_paths != [state.video_paths_list]:
            self.ego_mv_reader = MultiVideoReader([state.video_paths_list])
        bgr_frame = self.ego_mv_reader[frame_idx][0]
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
        if state.rrd_save_path is not None:
            recording.save(state.rrd_save_path)
            print(f"[Rerun] Logging to {state.rrd_save_path}")
        stream: rr.BinaryStream = recording.binary_stream()
        time_ns: Int[ndarray, "n_frames"] = state.shortest_timestamps

        for ts_idx, ts in tqdm(enumerate(time_ns), total=len(time_ns), desc="Processing frames"):
            rr.set_time(state.rr_log_paths.timeline, duration=ts * 1e-9, recording=recording)
            if self.ego_mv_reader is not None:
                ego_bgr_list: list[UInt8[ndarray, "h w 3"]] = self.ego_mv_reader[ts_idx]
                self._process_video_reader(
                    bgr_frames=ego_bgr_list,
                    video_log_paths=state.rr_log_paths.ego_video_log_paths,
                    recording=recording,
                )
                yield stream.read(), state

            if self.exo_mv_reader is not None:
                exo_bgr_list: list[UInt8[ndarray, "h w 3"]] = self.exo_mv_reader[ts_idx]
                exo_keypoints: dict[Path, dict[Handedness, Float[np.ndarray, "n_kpts 3"]]] = self._process_video_reader(
                    bgr_frames=exo_bgr_list,
                    video_log_paths=state.rr_log_paths.exo_video_log_paths,
                    recording=recording,
                    collect_keypoints=True,
                )
                if (
                    exo_keypoints
                    and state.current_prediction is not None
                    and state.current_prediction.pinhole_params_list is not None
                    and state.rr_log_paths.exo_video_log_paths is not None
                ):
                    self._triangulate_exo_views(
                        hand_keypoints=exo_keypoints,
                        pinhole_params_list=state.current_prediction.pinhole_params_list,
                        video_log_paths=state.rr_log_paths.exo_video_log_paths,
                        recording=recording,
                        parent_log_path=state.rr_log_paths.parent_log_path,
                    )
                yield stream.read(), state

    def calibrate_mv(self, state: AppState, rgb_list: list[UInt8[ndarray, "H W 3"]]) -> MVCalibResults:
        recording: rr.RecordingStream = get_recording(state.recording_id)
        return self.mv_calibrator(rgb_list=rgb_list, recording=recording)


def time_to_frame_idx(time_ns: int, frame_timestamps_ns: np.ndarray) -> int:
    """Map a timestamp (ns) to the closest frame idx at-or-before that time.

    The mapping mirrors how VideoFrameReference columns are generated from
    AssetVideo.read_frame_timestamps_nanos().
    """

    idx: int = int(np.searchsorted(frame_timestamps_ns, time_ns, side="right") - 1)
    return max(0, min(idx, len(frame_timestamps_ns) - 1))
