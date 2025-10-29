from dataclasses import dataclass
from pathlib import Path
from typing import Literal, TypedDict

import cv2
import numpy as np
import torch
from huggingface_hub import hf_hub_download
from jaxtyping import Float, Int, Num, UInt8
from numpy import ndarray
from rtmlib.tools.pose_estimation.rtmpose import RTMPose
from serde import from_dict, serde
from skimage.filters import gaussian
from wilor_nano.models.refinement_net import RefineNetOutput
from wilor_nano.models.wilor import WiLor
from wilor_nano.utils import utils


@dataclass
class HandKeypointDetectorConfig:
    verbose: bool
    hf_wilor_repo_id: str = "pablovela5620/wilor-nano"
    focal_length: int = 5000
    image_size: int = 256


class RefineNetNPOutput(TypedDict):
    """
    TypedDict for the output of RefineNet.

    This defines the structure and types of the dictionary returned by the RefineNet forward method.
    """

    global_orient: Float[torch.Tensor, "batch"]
    hand_pose: Float[torch.Tensor, "batch 15 3"]
    betas: Float[torch.Tensor, "batch 10"]
    pred_cam: Float[torch.Tensor, "batch 3"]


@serde
class RawWilorPred:
    """WiLor/RefineNet raw predictions for one hand (batch=1), model space."""

    global_orient: Float[ndarray, "batch=1 1 3"]
    """Root wrist global orientation (axis–angle, radians), shape (1, 1, 3)."""
    hand_pose: Float[ndarray, "batch=1 15 3"]
    """Local joint rotations for 15 MANO joints (axis–angle), shape (1, 15, 3)."""
    betas: Float[ndarray, "batch=1 10"]
    """MANO shape coefficients, shape (1, 10)."""
    pred_cam: Float[ndarray, "batch=1 3"]
    """Weak‑perspective camera in crop coords [s, tx, ty], shape (1, 3)."""
    pred_keypoints_3d: Float[ndarray, "batch=1 n_joints=21 3"]
    """21 MANO joints in model space, shape (1, 21, 3)."""
    pred_vertices: Float[ndarray, "batch=1 n_verts=778 3"]
    """778 MANO mesh vertices in model space, shape (1, 778, 3)."""


@serde
class FinalWilorPred:
    """Per‑hand predictions post‑processed to a canonical right‑hand frame."""

    global_orient: Float[ndarray, "batch=1 1 3"]
    """Canonicalized root orientation (axis–angle), shape (1, 1, 3)."""
    hand_pose: Float[ndarray, "batch=1 15 3"]
    """Canonicalized joint rotations (axis–angle), shape (1, 15, 3)."""
    betas: Float[ndarray, "batch=1 10"]
    """MANO shape coefficients, shape (1, 10)."""
    pred_cam: Float[ndarray, "batch=1 3"]
    """Weak‑perspective cam [s, tx, ty] in crop coords (handedness‑corrected), shape (1, 3)."""
    pred_cam_t_full: Float[ndarray, "batch=1 3"]
    """Full‑image translation for perspective projection, shape (1, 3)."""
    scaled_focal_length: float
    """Focal length scaled from crop size to full image (scalar)."""
    pred_keypoints_3d: Float[ndarray, "batch=1 n_joints=21 3"]
    """Canonicalized 3D joints in model space, shape (1, 21, 3)."""
    pred_vertices: Float[ndarray, "batch=1 n_verts=778 3"]
    """Canonicalized MANO mesh vertices in model space, shape (1, 778, 3)."""
    pred_keypoints_2d: Float[ndarray, "batch=1 n_joints=21 2"]
    """2D pixel coordinates in the original image, shape (1, 21, 2)."""
    confidence_2d: Float[ndarray, "batch=1 n_joints=21"]
    """2D keypoint confidence scores, shape (1, 21)."""


class RTMPoseHandKeypointDetector:
    def __init__(
        self,
        cfg: HandKeypointDetectorConfig,
        *,
        device: Literal["cuda", "cpu"] | None = None,
    ) -> None:
        self.cfg: HandKeypointDetectorConfig = cfg
        self._device: Literal["cuda", "cpu"] = (
            device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.init_model()

    def init_model(self) -> None:
        url: str = "https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/rtmpose-m_simcc-hand5_pt-aic-coco_210e-256x256-74fb594_20230320.zip"
        pose_input_size: tuple[int, int] = (256, 256)
        self.hand_model = RTMPose(onnx_model=url, model_input_size=pose_input_size, device=self._device)

    def __call__(
        self, image: np.ndarray, xyxy: list[list[float]] = []
    ) -> tuple[Float[ndarray, "batch=1 n_kpts=21 2"], Float[ndarray, "batch=1 n_kpts=21"]]:
        rtmpose_output: tuple[Float[ndarray, "batch=1 n_kpts=21 2"], Float[ndarray, "batch=1 n_kpts=21"]] = (
            self.hand_model(image, bboxes=xyxy)
        )
        keypoints: Float[ndarray, "batch=1 n_kpts=21 2"] = rtmpose_output[0]
        scores: Float[ndarray, "batch=1 n_kpts=21"] = rtmpose_output[1]
        return keypoints, scores


class WilorHandKeypointDetector:
    def __init__(self, cfg: HandKeypointDetectorConfig) -> None:
        self.cfg: HandKeypointDetectorConfig = cfg
        self.init_model()

    def init_model(self) -> None:
        device_type: Literal["cuda", "cpu"] = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device_type)
        self.dtype = torch.float16 if device_type == "cuda" else torch.float32

        wilor_pretrained_dir: Path = Path(__file__).parent.parent.resolve()
        mano_mean_path: Path = wilor_pretrained_dir / "pretrained_models" / "mano_mean_params.npz"
        if not mano_mean_path.exists():
            hf_hub_download(
                repo_id=self.cfg.hf_wilor_repo_id,
                subfolder="pretrained_models",
                filename="mano_mean_params.npz",
                local_dir=wilor_pretrained_dir,
            )

        mano_model_path: Path = wilor_pretrained_dir / "pretrained_models" / "mano_clean" / "MANO_RIGHT.pkl"
        if not mano_model_path.exists():
            hf_hub_download(
                repo_id=self.cfg.hf_wilor_repo_id,
                subfolder="pretrained_models/mano_clean",
                filename="MANO_RIGHT.pkl",
                local_dir=wilor_pretrained_dir,
            )
        self.wilor_model = WiLor(
            mano_root_dir=mano_model_path.parent,
            mano_mean_path=mano_mean_path,
            focal_length=self.cfg.focal_length,
            image_size=self.cfg.image_size,
        )
        wilor_model_path: Path = wilor_pretrained_dir / "pretrained_models" / "wilor_final.ckpt"
        if not wilor_model_path.exists():
            hf_hub_download(
                repo_id=self.cfg.hf_wilor_repo_id,
                subfolder="pretrained_models",
                filename="wilor_final.ckpt",
                local_dir=wilor_pretrained_dir,
            )
        # wilor model for keypoint predictions, does not provide confidence values, so these are gotten from the rtmpose hand model
        state_dict: dict[str, torch.Tensor] = torch.load(wilor_model_path, map_location=self.device)["state_dict"]
        self.wilor_model.load_state_dict(state_dict, strict=False)
        self.wilor_model.eval()
        self.wilor_model.to(self.device, dtype=self.dtype)

        rtmpose_device: Literal["cuda", "cpu"] = "cuda" if self.device.type == "cuda" else "cpu"
        self.hand_confidence_model = RTMPoseHandKeypointDetector(self.cfg, device=rtmpose_device)

    @torch.no_grad()
    def __call__(
        self,
        rgb_hw3: UInt8[ndarray, "H W 3"],
        xyxy: Float[np.ndarray, "1 4"],
        handedness: Literal["left", "right"],
        rescale_factor: float = 2.5,
    ) -> FinalWilorPred:
        center: Float[ndarray, "1 2"] = (xyxy[:, 2:4] + xyxy[:, 0:2]) / 2.0
        scale: Float[ndarray, "1 2"] = rescale_factor * (xyxy[:, 2:4] - xyxy[:, 0:2])
        img_patches_list: list[ndarray] = []
        img_size: Int[ndarray, "2"] = np.array([rgb_hw3.shape[1], rgb_hw3.shape[0]])

        bbox_size: float = float(scale[0].max())
        patch_width: int = self.cfg.image_size
        patch_height: int = self.cfg.image_size
        flip: bool = handedness == "left"
        box_center: Float[ndarray, "2"] = center[0]

        cvimg: Float[np.ndarray, "h w 3"] = rgb_hw3.copy().astype(np.float32)
        # Blur image to avoid aliasing artifacts
        downsampling_factor: float = (bbox_size * 1.0) / patch_width
        downsampling_factor: float = downsampling_factor / 2.0
        if downsampling_factor > 1.1:
            cvimg = gaussian(cvimg, sigma=(downsampling_factor - 1) / 2, channel_axis=2, preserve_range=True)

        patch_output: tuple[Float[ndarray, "h=256 w=256 3"], Float[ndarray, "2 3"]] = utils.generate_image_patch_cv2(
            cvimg,
            int(box_center[0]),
            int(box_center[1]),
            int(bbox_size),
            int(bbox_size),
            int(patch_width),
            int(patch_height),
            flip,
            1.0,
            0,
            border_mode=cv2.BORDER_CONSTANT,
        )
        img_patch_cv: Float[ndarray, "h=256 w=256 3"] = patch_output[0]
        img_patches_list.append(img_patch_cv)
        img_patches_np: Num[np.ndarray, "n_dets=1 h=256 w=256 3"] = np.stack(img_patches_list)
        img_patches: Float[torch.Tensor, "n_dets=1 h=256 w=256 3"] = torch.from_numpy(img_patches_np).to(
            device=self.device, dtype=self.dtype
        )
        # Run the WiLor model: outputs are PyTorch tensors keyed by component name
        wilor_output_raw: RefineNetOutput = self.wilor_model(img_patches)

        # Optional shape/type sanity check (e.g., with a serde-typed container).
        # Converts tensors to numpy for validation/logging, does not alter wilor_output_raw.
        raw_wilor_preds: RawWilorPred = from_dict(
            RawWilorPred, {k: v.cpu().float().numpy() for k, v in wilor_output_raw.items()}
        )
        # estimate the confidence of the keypoints using the rtmpose hand model
        conf_tuple: tuple[Float[ndarray, "batch=1 n_kpts=21 2"], Float[ndarray, "batch=1 n_kpts=21"]] = (
            self.hand_confidence_model(rgb_hw3, xyxy.tolist())
        )

        confidence_2d: Float[ndarray, "batch=1 n_kpts=21"] = conf_tuple[1]

        final_preds: FinalWilorPred = self.post_process(
            raw_wilor_preds=raw_wilor_preds,
            handedness=handedness,
            box_center=box_center,
            bbox_size=bbox_size,
            img_size=img_size,
            confidence_2d=confidence_2d,
        )

        return final_preds

    def post_process(
        self,
        *,
        raw_wilor_preds: RawWilorPred,
        handedness: Literal["left", "right"],
        box_center: Float[ndarray, "2"],
        bbox_size: float,
        img_size: Int[ndarray, "2"],
        confidence_2d: Float[ndarray, "batch=1 n_kpts=21"],
    ) -> FinalWilorPred:
        # Unpack for clarity; do not mutate inputs
        pred_cam_in: Float[ndarray, "batch=1 3"] = raw_wilor_preds.pred_cam
        pred_kpts3d_in: Float[ndarray, "batch=1 n_joints=21 3"] = raw_wilor_preds.pred_keypoints_3d
        pred_verts_in: Float[ndarray, "batch=1 n_verts=778 3"] = raw_wilor_preds.pred_vertices
        global_orient_in: Float[ndarray, "batch=1 1 3"] = raw_wilor_preds.global_orient
        hand_pose_in: Float[ndarray, "batch=1 15 3"] = raw_wilor_preds.hand_pose

        # Adjust weak-perspective camera for handedness: cam = [s, tx, ty]
        sx: int = 1 if handedness == "right" else -1  # +1 for right hand, -1 for left hand
        pred_cam: Float[ndarray, "batch=1 3"] = pred_cam_in.copy()
        pred_cam[:, 1] = sx * pred_cam[:, 1]

        # Mirror to canonical frame for left hand
        if handedness == "left":
            pred_keypoints_3d: Float[ndarray, "batch=1 n_joints=21 3"] = pred_kpts3d_in.copy()
            pred_keypoints_3d[:, :, 0] = -pred_keypoints_3d[:, :, 0]
            pred_vertices: Float[ndarray, "batch=1 n_verts=778 3"] = pred_verts_in.copy()
            pred_vertices[:, :, 0] = -pred_vertices[:, :, 0]

            # Rotation vectors (axis-angle): [x, y, z] -> [x, -y, -z]
            global_orient: Float[ndarray, "batch=1 1 3"] = np.concatenate(
                (global_orient_in[:, :, 0:1], -global_orient_in[:, :, 1:3]), axis=-1
            )
            hand_pose: Float[ndarray, "batch=1 15 3"] = np.concatenate(
                (hand_pose_in[:, :, 0:1], -hand_pose_in[:, :, 1:3]), axis=-1
            )
        else:
            pred_keypoints_3d = pred_kpts3d_in.copy()
            pred_vertices = pred_verts_in.copy()
            global_orient = global_orient_in.copy()
            hand_pose = hand_pose_in.copy()

        # Scale focal length from model crop size (image_size) to full image resolution
        scaled_focal_length: float = self.cfg.focal_length / self.cfg.image_size * img_size.max()

        # Lift weak-perspective cam to full-image translation for projection to 2D
        pred_cam_t_full: Float[ndarray, "batch=1 3"] = utils.cam_crop_to_full(
            pred_cam, box_center[None], bbox_size, img_size[None], scaled_focal_length
        )

        # Project 3D joints to 2D pixel coordinates in the original image
        pred_keypoints_2d: Float[ndarray, "batch=1 n_joints=21 2"] = utils.perspective_projection(
            pred_keypoints_3d,
            translation=pred_cam_t_full,
            focal_length=np.array([scaled_focal_length] * 2)[None],
            camera_center=img_size[None] / 2,
        )

        return FinalWilorPred(
            global_orient=global_orient,
            hand_pose=hand_pose,
            betas=raw_wilor_preds.betas.copy(),
            pred_cam=pred_cam,
            pred_cam_t_full=pred_cam_t_full,
            scaled_focal_length=scaled_focal_length,
            pred_keypoints_3d=pred_keypoints_3d,
            pred_vertices=pred_vertices,
            pred_keypoints_2d=pred_keypoints_2d,
            confidence_2d=confidence_2d,
        )
