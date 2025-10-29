from typing import Literal

from jaxtyping import Float, UInt8
from numpy import ndarray

from annotation_example.hand_keypoints import FinalWilorPred, HandKeypointDetectorConfig, WilorHandKeypointDetector


class Engine:
    """Container for inference engines used by the labeling UI."""

    def __init__(self) -> None:
        self._hand_detector: WilorHandKeypointDetector | None = None

    def infer_hand_keypoints(
        self,
        *,
        rgb_hw3: UInt8[ndarray, "H W 3"],
        xyxy: Float[ndarray, "1 4"],
        handedness: Literal["left", "right"],
    ) -> FinalWilorPred:
        """Run WiLor hand keypoint inference on a single RGB frame crop."""

        if self._hand_detector is None:
            self._hand_detector = WilorHandKeypointDetector(cfg=HandKeypointDetectorConfig(verbose=False))

        return self._hand_detector(
            rgb_hw3=rgb_hw3,
            xyxy=xyxy,
            handedness=handedness,
        )
