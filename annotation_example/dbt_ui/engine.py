from dataclasses import replace

import cv2
import numpy as np
from jaxtyping import UInt8
from numpy import ndarray
from simplecv.video_io import MultiVideoReader
from wilor_nano.hand_detection import DetectionResult, HandDetector, HandDetectorConfig

# Use the get_recording that supports optional FileSink saving via `rrd_path`.
from annotation_example.dbt_ui.state import AppState, CurrentPrediction


class Engine:
    """Used to hold neural network engines."""

    def __init__(
        self,
    ):
        self.hand_keypoint_engine = HandDetector(HandDetectorConfig(verbose=False))
        self._mv_reader: MultiVideoReader | None = None

    @property
    def mv_reader(self) -> MultiVideoReader | None:
        return self._mv_reader

    @mv_reader.setter
    def mv_reader(self, value: MultiVideoReader | None) -> None:
        self._mv_reader = value

    def predict_xyxy(self, state: AppState) -> AppState:
        # Convert current_time_ns to frame index using the logged frame timestamps.
        if state.frame_timestamps_ns is None or state.video_path is None:
            raise RuntimeError("Video not loaded or frame timestamps missing in state.")

        frame_idx: int = time_to_frame_idx(state.current_time_ns, state.frame_timestamps_ns)

        # Read frame from the underlying video using MultiVideoReader (BGR)
        if self.mv_reader is None or self.mv_reader.video_paths != [state.video_path]:
            self.mv_reader = MultiVideoReader([state.video_path])
        bgr_frame = self.mv_reader[frame_idx][0]
        rgb_hw3: UInt8[ndarray, "h w 3"] = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)

        # Run your model on the current frame (example placeholder)
        det_result: DetectionResult = self.hand_keypoint_engine(rgb_hw3=rgb_hw3, hand_conf=0.3)
        # update the app state with the new prediction
        current_prediction: CurrentPrediction = (
            state.current_prediction if state.current_prediction is not None else CurrentPrediction()
        )
        current_prediction.detection_results = det_result

        state: AppState = replace(state, current_prediction=current_prediction)
        return state


def time_to_frame_idx(time_ns: int, frame_timestamps_ns: np.ndarray) -> int:
    """Map a timestamp (ns) to the closest frame idx at-or-before that time.

    The mapping mirrors how VideoFrameReference columns are generated from
    AssetVideo.read_frame_timestamps_nanos().
    """

    idx: int = int(np.searchsorted(frame_timestamps_ns, time_ns, side="right") - 1)
    return max(0, min(idx, len(frame_timestamps_ns) - 1))
