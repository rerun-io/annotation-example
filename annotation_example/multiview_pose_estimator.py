from dataclasses import dataclass
from typing import Literal

import numpy as np
import rerun as rr
from einops import rearrange
from jaxtyping import Bool, Float, Float32, Float64, Int, UInt8
from numpy import ndarray
from rtmlib import YOLOX, RTMPose
from simplecv.camera_parameters import PinholeParameters
from simplecv.data.skeleton.coco_133 import COCO_133_IDS, LEFT_HAND_IDX, RIGHT_HAND_IDX
from simplecv.ops.triangulate import batch_triangulate, projectN3
from simplecv.rerun_custom_types import Points2DWithConfidence, confidence_scores_to_rgb

from annotation_example.coco133_layers import COCO133_PREDICTION_LAYER_TO_PATH, Coco133AnnotationLayer
from annotation_example.hand_keypoints import FinalWilorPred, HandKeypointDetectorConfig, WilorHandKeypointDetector

WILOR_CONFIDENCE_THRESHOLD: float = 0.5
WILOR_BBOX_EXPANSION_RATIO: float = 0.25


def project_multiview(
    xyzc: Float[ndarray, "n_kpts 4"],
    Pall: Float[ndarray, "n_views 3 4"],
) -> Float[ndarray, "n_views n_kpts 3"]:
    """
    Projects 3D keypoints (with confidence) to 2D image coordinates for multiple views.
    Handles potential division by zero by outputting NaN for those keypoints.

    Args:
        xyzc: Array of 3D keypoints and confidence scores (n_kpts, 4).
              The 4th column is confidence, used for masking, not projection.
        Pall: Array of projection matrices for each view (n_views, 3, 4).

    Returns:
        Array of projected 2D keypoints (u, v, w) for each view (n_views, n_kpts, 3).
        The third component 'w' is the depth scale factor, multiplied by the original
        confidence mask (0 or 1). Keypoints with original confidence <= 0 will have w=0.
        Keypoints where perspective division involved division by near-zero w' will have
        NaN values for u and v.
    """
    n_kpts = xyzc.shape[0]
    # Store original confidences for masking later
    original_conf = xyzc[:, 3].copy()

    # --- Vectorized Version ---
    # 1. Prepare homogeneous coordinates for 3D points (n_kpts, 4)
    #    We use xyz coordinates and append 1.
    kp3d_h = np.hstack((xyzc[:, :3], np.ones((n_kpts, 1), dtype=xyzc.dtype)))  # Shape: (K, 4)

    # 2. Perform matrix multiplication for all views at once.
    #    Using einsum: 'vmn,kn->vkm'
    #    Pall (V, 3, 4), kp3d_h (K, 4) -> kp2d_h (V, K, 3)
    #    kp2d_h[v, k, :] contains the (u', v', w') for view v, keypoint k
    kp2d_h = np.einsum("vmn,kn->vkm", Pall, kp3d_h, optimize=True)  # Shape: (V, K, 3)

    # 3. Perform perspective division (handle division by zero by setting to NaN)
    #    Extract w' (the depth scale factor) -> Shape (V, K, 1)
    w_prime = kp2d_h[..., 2:3]  # Keep last dimension

    #    Define mask for valid divisions (where w' is not close to zero)
    valid_division_mask = np.abs(w_prime) > 1e-8  # Shape: (V, K, 1)

    #    Calculate u = u'/w', v = v'/w'. Suppress warnings for division by zero.
    with np.errstate(divide="ignore", invalid="ignore"):
        uv_raw_division = kp2d_h[..., :2] / w_prime  # Shape: (V, K, 2)

    #    Use the mask to keep valid results and set others to NaN.
    #    The mask (V, K, 1) broadcasts correctly against (V, K, 2).
    uv_normalized = np.where(valid_division_mask, uv_raw_division, np.nan)  # Shape: (V, K, 2)

    # 4. Combine normalized uv with the original w' (depth scale)
    #    Concatenate along the last axis.
    kp2ds = np.concatenate((uv_normalized, w_prime), axis=-1)  # Shape: (V, K, 3)

    # 5. Apply the original confidence mask to the 'w' component.
    #    Keypoints with confidence <= 0 should have their w component zeroed out.
    #    This uses the confidence stored *before* it was potentially modified.
    confidence_mask = (original_conf[:, None] > 0.0).astype(kp2ds.dtype)  # Shape: (K, 1)
    # Broadcast mask (K, 1) to (V, K, 1) and multiply element-wise with w' column
    kp2ds[..., 2:3] *= confidence_mask  # Modifies the last column in place

    return kp2ds


@dataclass
class MVHistory:
    """Keeps track of the previous two timesteps of 3D keypoints."""

    xyzc_t: Float32[ndarray, "n_kpts=133 4"] | None = None
    """current frame (t); ``None`` when unavailable."""
    xyzc_t1: Float32[ndarray, "n_kpts=133 4"] | None = None
    """previous frame (t-1); ``None`` when unavailable."""
    uvc_t: Float32[ndarray, "n_views n_kpts=133 3"] | None = None
    """Current per-view 2D detections (u, v, confidence)."""
    uvc_extrap: Float32[ndarray, "n_views n_kpts=133 3"] | None = None
    """Per-view 2D projections derived from the temporally extrapolated 3D keypoints."""


@dataclass(frozen=True, slots=True)
class ModelAssets:
    """Model assets for detector and pose estimators."""

    det: str
    """Download URL or local path to the YOLOX ONNX model."""
    det_input_size: tuple[int, int]
    """Input width-height pair expected by YOLOX."""
    pose: str
    """Download URL or local path to the RTMPose ONNX model."""
    pose_input_size: tuple[int, int]
    """Input width-height pair expected by RTMPose."""


MODE: dict[str, ModelAssets] = {
    "performance": ModelAssets(
        det="https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/yolox_x_8xb8-300e_humanart-a39d44ed.zip",
        det_input_size=(640, 640),
        pose="https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/rtmpose-x_simcc-body7_pt-body7_700e-384x288-71d7b7e9_20230629.zip",
        pose_input_size=(288, 384),
    ),
    "lightweight": ModelAssets(
        det="https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/yolox_tiny_8xb8-300e_humanart-6f3252f9.zip",
        det_input_size=(416, 416),
        pose="https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/rtmpose-s_simcc-body7_pt-body7_420e-256x192-acd4a1ef_20230504.zip",
        pose_input_size=(192, 256),
    ),
    "balanced": ModelAssets(
        det="https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/yolox_m_8xb8-300e_humanart-c2c7a14a.zip",
        det_input_size=(640, 640),
        pose="https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/rtmpose-m_simcc-body7_pt-body7_420e-256x192-e48f03d0_20230504.zip",
        pose_input_size=(192, 256),
    ),
    "wholebody": ModelAssets(
        det="https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/yolox_m_8xb8-300e_humanart-c2c7a14a.zip",
        det_input_size=(640, 640),
        pose="https://download.openmmlab.com/mmpose/v1/projects/rtmw/onnx_sdk/rtmw-dw-x-l_simcc-cocktail14_270e-256x192_20231122.zip",
        pose_input_size=(192, 256),
    ),
}


@dataclass
class MultiviewBodyTrackerConfig:
    """Configuration options for the multiview body tracker runtime."""

    mode: Literal["lightweight", "balanced", "performance", "wholebody"] = "wholebody"
    """Preset selecting detector and pose assets tuned for latency versus accuracy."""
    backend: Literal["onnxruntime"] = "onnxruntime"
    """Inference backend used to execute ONNX models."""
    device: Literal["cpu", "cuda"] = "cuda"
    """Hardware accelerator requested by the ONNX runtime backend."""
    keypoint_threshold: float = 0.7
    """Minimum 2D keypoint confidence required for a detection to be kept."""
    cams_for_detection_idx: list[int] | None = None
    """Subset of camera indices to run detection on; ``None`` evaluates every view."""
    use_wilor: bool = False
    """Whether to use Wilor-Nano for hand keypoints instead of RTMPose."""
    perform_tracking: bool = True
    """Whether to extrapolate poses from historical frames to assist detection."""
    verbose: bool = False
    """Enables additional debug logging when ``True``."""


class MultiviewBodyTracker:
    def __init__(
        self,
        config: MultiviewBodyTrackerConfig,
        filter_body_idxes: Int[ndarray, "idx"] | None = None,
    ) -> None:
        """Optional index list restricting which detected bodies propagate downstream."""
        self.config: MultiviewBodyTrackerConfig = config
        self.num_keypoints: int = 133  # Default number of keypoints for pose estimation
        self.filter_body_idxes: Int[ndarray, "idx"] = (
            filter_body_idxes if filter_body_idxes is not None else np.arange(self.num_keypoints, dtype=np.intp)
        )

        assets: ModelAssets = MODE[config.mode]
        pose: str = assets.pose
        pose_input_size: tuple[int, int] = assets.pose_input_size

        det: str = assets.det
        det_input_size: tuple[int, int] = assets.det_input_size
        self.det_model = YOLOX(det, model_input_size=det_input_size, backend=config.backend, device=config.device)
        self.pose_model = RTMPose(
            pose,
            model_input_size=pose_input_size,
            to_openpose=False,
            backend=config.backend,
            device=config.device,
        )
        self.keypoint_threshold: float = config.keypoint_threshold
        self.hand_keypoint_engine: WilorHandKeypointDetector | None = None
        if self.config.use_wilor:
            self.hand_keypoint_engine = WilorHandKeypointDetector(HandKeypointDetectorConfig(verbose=False))

        self.tracked: bool = False

    def __call__(
        self,
        *,
        bgr_list: list[UInt8[ndarray, "H W 3"]],
        pinhole_list: list[PinholeParameters],
        pred_state: MVHistory,
        recording: rr.RecordingStream | None = None,
    ) -> MVHistory:
        """
        Assumes to be only one person in the image, take the highest score person
        """
        # filter out the cameras that are not used for detection if provided
        Pall: Float32[ndarray, "n_views 3 4"] = np.array(
            [pinhole.projection_matrix for pinhole in pinhole_list], dtype=np.float32
        )
        if self.config.cams_for_detection_idx is not None:
            Pall: Float32[ndarray, "n_views 3 4"] = np.array([Pall[i] for i in self.config.cams_for_detection_idx])

        ##################################################################################
        # if we have keypoints from previous two timesteps, then use them to extrapolate #
        # the 3d keypoints which we can project to 2d and use for detection              #
        ##################################################################################
        xyzc_extrap: Float32[ndarray, "n_kpts 4"] | None = None
        bboxes_extrap: Float32[ndarray, "n_views 4"] | None = None
        tracked_confidences: Float32[ndarray, "n_kpts"] | None = None
        pred_state.uvc_extrap = None
        if (pred_state.xyzc_t1 is not None and pred_state.xyzc_t is not None) and self.config.perform_tracking:
            xyzc_extrap = self.extrapolate_3d_keypoints(xyzc_t=pred_state.xyzc_t, xyzc_t1=pred_state.xyzc_t1)
            # project the extrapolated 3d keypoints to 2d
            uvc_extrap_float: Float[ndarray, "n_views n_kpts 3"] = projectN3(xyzc_extrap, Pall)
            uvc_extrap: Float32[ndarray, "n_views n_kpts 3"] = np.asarray(uvc_extrap_float, dtype=np.float32)
            pred_state.uvc_extrap = uvc_extrap
            tracked_confidences = np.clip(xyzc_extrap[:, 3].astype(np.float32, copy=True), 0.0, 1.0)

            uv_max: Float[ndarray, "n_views 2"] = np.nanmax(uvc_extrap[:, :, 0:2], axis=1)
            uv_min: Float[ndarray, "n_views 2"] = np.nanmin(uvc_extrap[:, :, 0:2], axis=1)
            bboxes_extrap = np.concatenate([uv_min, uv_max], axis=1).astype(np.float32)

        uvc_list: list[Float32[ndarray, "n_kpts 3"]] = []
        for image_idx, bgr in enumerate(bgr_list):
            if xyzc_extrap is not None and bboxes_extrap is not None:
                bboxes: Float32[ndarray, "n_dets 4"] = rearrange(bboxes_extrap[image_idx], "B -> 1 B")
            else:
                det_output: np.ndarray | tuple[np.ndarray, ...] = self.det_model(bgr)
                if isinstance(det_output, tuple):
                    det_bboxes_np: np.ndarray = det_output[0]
                else:
                    det_bboxes_np = det_output
                bboxes: Float32[ndarray, "n_dets 4"] = np.asarray(det_bboxes_np, dtype=np.float32)

            match bboxes.shape[0]:
                # No detections, set outputs to None for this view
                case 0:
                    uvc_list.append(np.zeros((self.num_keypoints, 3)).astype(np.float32))
                case _:
                    # Select the detection with the highest score (assumes bboxes are sorted by score).
                    # If not sorted, you should select the bbox with the highest score here.
                    bboxes: Float32[ndarray, "1 4"] = bboxes[0:1]

                    bbox_list: list[list[float]] = bboxes.astype(np.float32).tolist()
                    pose_output: tuple[
                        Float64[ndarray, "n_dets n_kpts=133 2"], Float32[ndarray, "n_dets n_kpts=133"]
                    ] = self.pose_model(bgr, bboxes=bbox_list)
                    keypoints: Float64[ndarray, "n_dets n_kpts=133 2"] = pose_output[0]
                    scores: Float32[ndarray, "n_dets n_kpts=133"] = pose_output[1]

                    filtered_keypoints, filtered_scores = self.filter_kpt_outputs(keypoints.astype(np.float32), scores)
                    # filtered_scores = filtered_scores.astype(np.float32, copy=True)
                    # filtered_scores[filtered_scores < self.config.keypoint_threshold] = 0.0
                    # filtered_scores[filtered_scores >= self.config.keypoint_threshold] = 1.0

                    if self.config.use_wilor:
                        filtered_keypoints = self._refine_hand_keypoints_with_wilor(
                            bgr=bgr,
                            keypoints=filtered_keypoints,
                            confidences=filtered_scores,
                        )

                    if self.config.verbose:
                        self._log_uvc_layer(
                            view_idx=image_idx,
                            pinhole=pinhole_list[image_idx],
                            keypoints=filtered_keypoints,
                            confidences=filtered_scores,
                            layer=Coco133AnnotationLayer.RAW_2D,
                            mask_below_threshold=True,
                            recording=recording,
                        )
                        if tracked_confidences is not None and pred_state.uvc_extrap is not None:
                            tracked_uv: Float32[ndarray, "n_kpts 2"] = pred_state.uvc_extrap[image_idx, :, 0:2]
                            self._log_uvc_layer(
                                view_idx=image_idx,
                                pinhole=pinhole_list[image_idx],
                                keypoints=tracked_uv,
                                confidences=tracked_confidences,
                                layer=Coco133AnnotationLayer.TRACKED_2D,
                                mask_below_threshold=False,
                                recording=recording,
                            )

                    # can't be nan as it messes with triangulation
                    # filtered_scores[filtered_scores < self.keypoint_threshold] = 0
                    # filtered_scores[filtered_scores >= self.keypoint_threshold] = 1
                    uvc: Float32[ndarray, "n_kpts 3"] = np.concatenate(
                        [filtered_keypoints, filtered_scores[:, None]], axis=1
                    )
                    uvc_list.append(uvc)

        multiview_uvc: Float32[ndarray, "n_views n_kpts 3"] = np.stack(uvc_list).astype(np.float32)
        pred_state.uvc_t = multiview_uvc
        xyzc: Float64[ndarray, "n_kpts 4"] = batch_triangulate(
            keypoints_2d=multiview_uvc,
            projection_matrices=Pall,
            min_views=2,
        )

        filtered_xyzc = xyzc[self.filter_body_idxes]
        # check if more than half the keypoints are below the threshold, if so, don't track
        if np.sum(filtered_xyzc[:, 3] > self.config.keypoint_threshold) < filtered_xyzc.shape[0] / 2:
            pred_state.xyzc_t = None
            pred_state.xyzc_t1 = None

        pred_state.xyzc_t1 = pred_state.xyzc_t
        pred_state.xyzc_t = xyzc.astype(np.float32)

        return pred_state

    def _log_uvc_layer(
        self,
        *,
        view_idx: int,
        pinhole: PinholeParameters,
        keypoints: Float32[ndarray, "n_kpts 2"],
        confidences: Float32[ndarray, "n_kpts"],
        layer: Coco133AnnotationLayer,
        mask_below_threshold: bool,
        recording: rr.RecordingStream | None,
    ) -> None:
        """Log a COCO-133 2D layer for a single view."""

        try:
            view_name_attr: str | None = pinhole.name  # type: ignore[attr-defined]
        except AttributeError:
            view_name_attr = None
        view_name: str = view_name_attr if view_name_attr else f"view_{view_idx}"

        filtered_keypoints: Float32[ndarray, "n_kpts 2"]
        filtered_confidences: Float32[ndarray, "n_kpts"]
        if mask_below_threshold:
            min_conf: float = float(self.config.keypoint_threshold)
            visibility_mask: Bool[ndarray, "n_kpts"] = confidences >= min_conf
            filtered_keypoints = np.where(
                visibility_mask[:, None],
                keypoints,
                np.nan,
            ).astype(np.float32, copy=False)
            filtered_confidences = np.where(
                visibility_mask,
                confidences,
                0.0,
            ).astype(np.float32, copy=False)
        else:
            filtered_keypoints = keypoints.astype(np.float32, copy=False)
            filtered_confidences = confidences.astype(np.float32, copy=False)

        finite_mask: Bool[ndarray, "n_kpts"] = np.isfinite(filtered_keypoints[:, 0]) & np.isfinite(
            filtered_keypoints[:, 1]
        )
        filtered_keypoints = np.where(finite_mask[:, None], filtered_keypoints, np.nan)
        filtered_confidences = np.where(finite_mask, filtered_confidences, 0.0)

        confidence_rgb: UInt8[ndarray, "n_kpts 3"] = confidence_scores_to_rgb(
            filtered_confidences[np.newaxis, :, np.newaxis]
        )[0]
        layer_segment: str = COCO133_PREDICTION_LAYER_TO_PATH[layer]

        rr.log(
            f"/world/exo/{view_name}/pinhole/pred/coco133_uv/{layer_segment}",
            Points2DWithConfidence(
                positions=filtered_keypoints,
                confidences=filtered_confidences,
                class_ids=int(layer),
                keypoint_ids=COCO_133_IDS,
                show_labels=False,
                colors=confidence_rgb,
            ),
            recording=recording,
        )

    def _refine_hand_keypoints_with_wilor(
        self,
        *,
        bgr: UInt8[ndarray, "H W 3"],
        keypoints: Float32[ndarray, "n_kpts 2"],
        confidences: Float32[ndarray, "n_kpts"],
    ) -> Float32[ndarray, "n_kpts 2"]:
        engine = self.hand_keypoint_engine
        if engine is None:
            return keypoints

        refined_keypoints: Float32[ndarray, "n_kpts 2"] = keypoints.copy()
        height: int = bgr.shape[0]
        width: int = bgr.shape[1]
        rgb_hw3: UInt8[ndarray, "H W 3"] = bgr[..., ::-1]
        score_threshold: float = max(self.config.keypoint_threshold, WILOR_CONFIDENCE_THRESHOLD)

        for hand_indices_raw, handedness in ((LEFT_HAND_IDX, "left"), (RIGHT_HAND_IDX, "right")):
            hand_indices: np.ndarray = np.asarray(hand_indices_raw, dtype=np.intp)
            hand_uv: Float32[ndarray, "hand 2"] = refined_keypoints[hand_indices, :].copy()
            hand_scores: Float32[ndarray, "hand"] = confidences[hand_indices]

            low_conf_mask: np.ndarray = np.asarray(hand_scores < score_threshold, dtype=bool)
            hand_uv[low_conf_mask, :] = np.nan

            bbox: Float32[ndarray, "4"] | None = self._compute_hand_bbox(
                hand_uv=hand_uv,
                image_shape=(height, width),
                expansion_ratio=WILOR_BBOX_EXPANSION_RATIO,
            )
            if bbox is None:
                continue

            xyxy: Float32[ndarray, "1 4"] = bbox[np.newaxis, :]
            wilor_pred: FinalWilorPred = engine(rgb_hw3=rgb_hw3, xyxy=xyxy, handedness=handedness)
            pred_uv: Float32[ndarray, "1 21 2"] = wilor_pred.pred_keypoints_2d.astype(np.float32, copy=False)

            refined_hand_uv: Float32[ndarray, "21 2"] = pred_uv[0]
            refined_hand_uv[:, 0] = np.clip(refined_hand_uv[:, 0], 0.0, float(width - 1))
            refined_hand_uv[:, 1] = np.clip(refined_hand_uv[:, 1], 0.0, float(height - 1))

            refined_keypoints[hand_indices, :] = refined_hand_uv

        return refined_keypoints

    @staticmethod
    def _compute_hand_bbox(
        *,
        hand_uv: Float32[ndarray, "hand 2"],
        image_shape: tuple[int, int],
        expansion_ratio: float,
    ) -> Float32[ndarray, "4"] | None:
        valid_mask: np.ndarray = np.isfinite(hand_uv[:, 0]) & np.isfinite(hand_uv[:, 1])
        if not bool(np.any(valid_mask)):
            return None

        valid_uv: Float32[ndarray, "valid 2"] = hand_uv[valid_mask, :]
        min_xy: Float32[ndarray, "2"] = np.nanmin(valid_uv, axis=0).astype(np.float32, copy=False)
        max_xy: Float32[ndarray, "2"] = np.nanmax(valid_uv, axis=0).astype(np.float32, copy=False)

        side_length: float = float(max(max_xy[0] - min_xy[0], max_xy[1] - min_xy[1]))
        if not np.isfinite(side_length) or side_length <= 0.0:
            return None

        center_xy: Float32[ndarray, "2"] = ((min_xy + max_xy) * 0.5).astype(np.float32, copy=False)
        half_side: float = 0.5 * side_length * (1.0 + expansion_ratio)
        if half_side <= 0.0:
            return None

        x1: float = float(center_xy[0] - half_side)
        y1: float = float(center_xy[1] - half_side)
        x2: float = float(center_xy[0] + half_side)
        y2: float = float(center_xy[1] + half_side)

        height, width = image_shape
        x1 = float(np.clip(x1, 0.0, float(width - 1)))
        x2 = float(np.clip(x2, 0.0, float(width - 1)))
        y1 = float(np.clip(y1, 0.0, float(height - 1)))
        y2 = float(np.clip(y2, 0.0, float(height - 1)))

        if x2 <= x1 or y2 <= y1:
            return None

        bbox: Float32[ndarray, "4"] = np.array([x1, y1, x2, y2], dtype=np.float32)
        return bbox

    def extrapolate_3d_keypoints(
        self, xyzc_t: Float32[ndarray, "n_kpts 4"], xyzc_t1: Float32[ndarray, "n_kpts 4"]
    ) -> Float32[ndarray, "n_kpts 4"]:
        """
        Extrapolates 3D keypoints with confidence scores using data from
        the previous two timesteps (t-1 and t-2).
        """
        # extrapolate 3d keypoints from the previous frames
        xyzc_extrap = 2 * xyzc_t - xyzc_t1
        return xyzc_extrap

    def filter_kpt_outputs(
        self,
        keypoints: Float32[ndarray, "n_dets n_kpts 2"],
        scores: Float32[ndarray, "n_dets n_kpts"],
    ) -> tuple[Float32[ndarray, "n_kpts 2"], Float32[ndarray, "n_kpts"]]:
        """
        Filter keypoints based on the highest score
        """
        max_scores: Float32[ndarray, "n_dets"] = scores.max(axis=1)
        max_idx = max_scores.argmax()

        return keypoints[max_idx], scores[max_idx]
