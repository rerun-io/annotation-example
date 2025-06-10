import os
import random
import re
import uuid
from dataclasses import dataclass, fields
from enum import Enum, auto
from pathlib import Path
from typing import Literal

import cv2
import gradio as gr
import numpy as np
import rerun as rr
import rerun.blueprint as rrb
import torch
from einops import rearrange
from gradio_rerun import Rerun
from jaxtyping import Bool, Float, Float32, Float64, UInt8
from monopriors.depth_utils import depth_edges_mask, depth_to_points
from monopriors.relative_depth_models import (
    RelativeDepthPrediction,
    get_relative_predictor,
)
from monopriors.relative_depth_models.base_relative_depth import BaseRelativePredictor
from numpy import ndarray
from PIL import Image
from rtmlib import RTMPose
from serde import serde
from serde.json import from_json, to_json
from simplecv.apis.convert_to_rrd import confidence_scores_to_rgb
from simplecv.camera_parameters import Extrinsics, Intrinsics, PinholeParameters
from simplecv.rerun_log_utils import Points2DWithConfidence
from torchvision import transforms
from transformers import AutoModelForImageSegmentation

from annotation_example.gradio_ui.person_annot_callbacks import RerunLogPaths, single_view_update_keypoints
from annotation_example.gradio_ui.person_annot_utils import (
    SELECTED_COLOR,
    SELECTED_XYXY_COLOR,
    CurrentState,
    SAM2KeypointContainer,
    SVPrediction,
    XYXYContainer,
    get_recording,
)
from annotation_example.skeletons import COCO_17_ID2NAME, COCO_17_IDS, COCO_17_LINKS, COCO_17_NAME2ID


@serde
class LabelInfo:
    skip: list[str] = None
    annotated: list[str] = None


def natural_sort_key(path: Path):
    # Regex to capture the base ID and the page number
    # It looks for a pattern like "some_id_pNUMBER.extension"
    # It handles cases where there might be multiple underscores before "_p"
    match = re.match(r"^(.*?)_p(\d+)\..*$", path.name)
    if match:
        base_id = match.group(1)
        page_number = int(match.group(2))
        return (base_id, page_number, path.suffix)
    else:
        # Fallback for filenames that don't match the pattern
        return (path.stem, 0, path.suffix)  # Sort unmatched names first or as a group


def get_image_list(img_dir: Path, annotation_info: LabelInfo) -> list[Path]:
    assert img_dir.exists(), "image_dir does not exist. Check the environment variable under [tool.pixi.activation]"
    img_paths_list: list[Path] = [f for f in img_dir.glob("*") if f.is_file()]
    # get 100 random images from the directory
    img_paths_list = [f for f in img_paths_list if f.suffix.lower() in [".jpg", ".jpeg", ".png", ".webp"]]

    img_paths_list.sort(key=natural_sort_key)  # Sort by base ID and page number, then reverse order
    filtered_names = set(annotation_info.skip)
    filtered_img_paths_list: list[Path] = [
        img_path for img_path in img_paths_list if img_path.stem not in filtered_names
    ]
    return filtered_img_paths_list


img_dir_env = os.getenv("IMAGES_TO_ANNOTATE_DIR")
if img_dir_env is None:
    raise OSError("IMAGES_TO_ANNOTATE_DIR environment variable not set.")
img_dir = Path(img_dir_env)

annot_save_dir: Path = Path("data") / f"{img_dir.name}-annotations"
rrd_dir: Path = annot_save_dir / "rrd_files"
rrd_dir.mkdir(parents=True, exist_ok=True)
label_info_json_path: Path = annot_save_dir / "label_info.json"

# Load existing annotations if available, else create a new one
try:
    LABEL_INFO: LabelInfo = from_json(LabelInfo, label_info_json_path.read_text())
except FileNotFoundError:
    LABEL_INFO = LabelInfo(skip=[], annotated=[])

EXAMPLE_IMAGES: list[Path] = get_image_list(img_dir, annotation_info=LABEL_INFO)

if gr.NO_RELOAD:
    kpt_only_model = RTMPose(
        "https://download.openmmlab.com/mmpose/v1/projects/rtmw/onnx_sdk/rtmw-dw-x-l_simcc-cocktail14_270e-384x288_20231122.zip",
        model_input_size=(288, 384),
        to_openpose=False,
        backend="onnxruntime",
        device="cuda",
    )
    depth_predictor: BaseRelativePredictor = get_relative_predictor("MogeV1Predictor")(device="cuda")

    torch.set_float32_matmul_precision(["high", "highest"][0])

    birefnet = AutoModelForImageSegmentation.from_pretrained("ZhengPeng7/BiRefNet", trust_remote_code=True)
    birefnet.to("cuda")

    transform_image = transforms.Compose(
        [
            transforms.Resize((1024, 1024)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )

    # import torch
    # from hydra import initialize_config_dir
    # from hydra.core.global_hydra import GlobalHydra
    # from sam2.build_sam import build_sam2
    # from sam2.sam2_image_predictor import SAM2ImagePredictor

    # project_root = Path.cwd()
    # print(f"Project root: {project_root}")

    # checkpoints_dir = project_root / "checkpoints"

    # # Config for SAM2 model
    # # The config file name (without .yaml extension)
    # sam2_config_name = "edgetam"
    # # Full path to the .yaml file
    # sam2_config_yaml_file = checkpoints_dir / f"{sam2_config_name}.yaml"
    # # Full path to the checkpoint (.pt) file
    # sam2_checkpoint_pt_file = checkpoints_dir / "edgetam.pt"

    # assert sam2_config_yaml_file.exists(), f"SAM2 Model config file {sam2_config_yaml_file} does not exist."
    # assert sam2_checkpoint_pt_file.exists(), f"SAM2 Model checkpoint file {sam2_checkpoint_pt_file} does not exist."

    # # Initialize Hydra to add our custom config directory (`checkpoints_dir`) to its search path.

    # # Clear any existing Hydra instance. This is important if Hydra was initialized elsewhere.
    # if GlobalHydra.instance().is_initialized():
    #     GlobalHydra.instance().clear()

    # # The `initialize_config_dir` context manager sets up Hydra's search path.
    # # `config_dir` should be the absolute path to the directory containing 'edgetam.yaml'.
    # with initialize_config_dir(config_dir=str(checkpoints_dir.resolve()), job_name="sam2_custom_config_loading"):
    #     # Now, when `build_sam2` is called with `sam2_config_name` (e.g., "edgetam"),
    #     # Hydra's `compose` function (used inside `build_sam2`) will look for "edgetam.yaml"
    #     # in the `checkpoints_dir` we just added to the search path.
    #     # The checkpoint path should be an absolute path string.
    #     predictor = SAM2ImagePredictor(
    #         build_sam2(config_file=sam2_config_name, ckpt_path=str(sam2_checkpoint_pt_file.resolve()), device="cuda")
    #     )


def new_annotation_context(rec) -> None:
    rec.log(
        "/",
        rr.AnnotationContext(
            [
                rr.ClassDescription(
                    info=rr.AnnotationInfo(id=2, label="Wholebody 133", color=(0, 0, 255)),
                    keypoint_annotations=[rr.AnnotationInfo(id=id, label=name) for id, name in COCO_17_ID2NAME.items()],
                    keypoint_connections=COCO_17_LINKS,
                ),
                rr.AnnotationInfo(id=0, label="Background"),
                rr.AnnotationInfo(id=1, label="Person", color=(0, 0, 0)),
            ]
        ),
        static=True,
    )


def remove_bg(rgb: Image.Image) -> Image.Image:
    """
    Apply BiRefNet-based image segmentation to remove the background.

    This function preprocesses the input image, runs it through a BiRefNet segmentation model to obtain a mask,
    and applies the mask as an alpha (transparency) channel to the original image.

    Args:
        image (PIL.Image): The input RGB image.

    Returns:
        PIL.Image: The image with the background removed, using the segmentation mask as transparency.
    """
    image_size = rgb.size
    input_images = transform_image(rgb).unsqueeze(0).to("cuda")
    # Prediction
    with torch.no_grad():
        preds = birefnet(input_images)[-1].sigmoid().cpu()
    pred = preds[0].squeeze()
    pred_pil = transforms.ToPILImage()(pred)
    mask = pred_pil.resize(image_size)
    rgb.putalpha(mask)
    return rgb


def log_relative_pred(
    rec: rr.RecordingStream,
    parent_log_path: Path,
    relative_pred: RelativeDepthPrediction,
    rgb_hw3: UInt8[np.ndarray, "h w 3"],
    remove_flying_pixels: bool = True,
    depth_edge_threshold: int | float = 0.5,
) -> Float32[np.ndarray, "h w"]:
    cam_log_path: Path = parent_log_path / "camera"
    pinhole_path: Path = cam_log_path / "pinhole"

    # assume camera is at the origin
    cam_T_world_44: Float64[np.ndarray, "4 4"] = np.eye(4)

    rec.log(
        f"{cam_log_path}",
        rr.Transform3D(
            translation=cam_T_world_44[:3, 3],
            mat3x3=cam_T_world_44[:3, :3],
            from_parent=True,
        ),
    )
    rec.log(
        f"{pinhole_path}",
        rr.Pinhole(
            image_from_camera=relative_pred.K_33,
            width=rgb_hw3.shape[1],
            height=rgb_hw3.shape[0],
            camera_xyz=rr.ViewCoordinates.RDF,
        ),
    )

    depth_hw: Float32[np.ndarray, "h w"] = relative_pred.depth
    # set any inf values to nan
    depth_hw[np.isinf(depth_hw)] = np.nan
    if remove_flying_pixels:
        edges_mask: Bool[np.ndarray, "h w"] = depth_edges_mask(depth_hw, threshold=depth_edge_threshold)
        depth_hw: Float32[np.ndarray, "h w"] = depth_hw * ~edges_mask

    rec.log(f"{pinhole_path}/depth", rr.DepthImage(depth_hw))

    depth_1hw: Float32[np.ndarray, "1 h w"] = rearrange(depth_hw, "h w -> 1 h w")
    pts_3d: Float32[np.ndarray, "h w 3"] = depth_to_points(depth_1hw, relative_pred.K_33)

    rec.log(
        f"{parent_log_path}/point_cloud",
        rr.Points3D(
            positions=pts_3d.reshape(-1, 3),
            colors=rgb_hw3.reshape(-1, 3),
        ),
    )

    return depth_hw


def log_sv_pred(
    rec: rr.RecordingStream,
    state: CurrentState,
    sv_prediction: SVPrediction,
) -> None:
    blueprint = create_blueprint(state)
    rr_log_paths: RerunLogPaths = state.rerun_log_paths

    rec.set_time("iteration", sequence=0)
    rec.log("/", rr.ViewCoordinates.RDF, static=True)
    new_annotation_context(rec)
    rec.send_blueprint(blueprint)

    # log info message
    rec.log(
        "info",
        rr.TextDocument(
            text=sv_prediction.info_message,
            media_type="markdown",
        ),
    )

    cam_T_world_44: Float[np.ndarray, "4 4"] = sv_prediction.pinhole_params.extrinsics.cam_T_world
    K_33: Float[np.ndarray, "3 3"] = sv_prediction.pinhole_params.intrinsics.k_matrix.astype(np.float32)
    rec.log(
        f"{rr_log_paths['camera_log_path']}",
        rr.Transform3D(
            translation=cam_T_world_44[:3, 3],
            mat3x3=cam_T_world_44[:3, :3],
            from_parent=True,
        ),
    )
    rec.log(
        f"{rr_log_paths['pinhole_log_path']}",
        rr.Pinhole(
            image_from_camera=K_33,
            width=sv_prediction.pinhole_params.intrinsics.width,
            height=sv_prediction.pinhole_params.intrinsics.height,
            camera_xyz=rr.ViewCoordinates.RDF,
        ),
    )

    segmented_rgb: UInt8[np.ndarray, "H W 3"] = sv_prediction.rgb_hw3.copy()
    if sv_prediction.mask is not None:
        # Set RGB values to black [0, 0, 0] for these transparent areas.
        # This is done before passing to the depth predictor.
        transparent_mask: Bool[np.ndarray, "H W"] = sv_prediction.mask
        segmented_rgb[~transparent_mask] = [0, 0, 0]

    rec.log(f"{rr_log_paths['original_image_path']}", rr.Image(sv_prediction.rgb_hw3, rr.ColorModel.RGB).compress())
    rec.log(
        f"{rr_log_paths['annotated_image_path']}",
        rr.Image(segmented_rgb, rr.ColorModel.RGB).compress(),
    )
    rec.log(
        f"{rr_log_paths['annotated_image_path']}/segmentation",
        rr.SegmentationImage(sv_prediction.mask.astype(np.uint8)),
    )

    rec.log(f"{rr_log_paths['pinhole_log_path']}/depth", rr.DepthImage(sv_prediction.depth_relative))

    depth_1hw: Float[np.ndarray, "1 h w"] = rearrange(sv_prediction.depth_relative, "h w -> 1 h w")
    pts_3d: Float[np.ndarray, "h w 3"] = depth_to_points(depth_1hw, K_33)

    rec.log(
        f"{rr_log_paths['parent_log_path']}/point_cloud",
        rr.Points3D(
            positions=pts_3d.reshape(-1, 3),
            colors=sv_prediction.rgb_hw3.reshape(-1, 3),
        ),
    )

    # log bbox
    xyxy: XYXYContainer
    for idx, xyxy in enumerate(sv_prediction.xyxy_list):
        # update visualization for all bounding boxes
        rec.log(
            f"{state.rerun_log_paths['annotated_image_path']}/bbox_{idx}",
            rr.Boxes2D(
                array=xyxy.bbox,
                array_format=rr.Box2DFormat.XYXY,
                labels=[f"Person {idx}"],
                show_labels=True,
            ),
        )
    # log keypoints
    current_uvc: Float[ndarray, "n_kpts 3"]
    for idx, current_uvc in enumerate(sv_prediction.uvc_list):
        current_uv: Float[ndarray, "n_kpts 2"] = current_uvc[:, 0:2]  # Extract x, y coordinates
        scores: Float[ndarray, "n_kpts"] = current_uvc[:, -1].astype(np.float32)  # Placeholder for scores
        colors: UInt8[np.ndarray, "n_kpts 3"] = confidence_scores_to_rgb(
            confidence_scores=scores[np.newaxis, :, np.newaxis]
        ).squeeze()
        rec.log(
            f"{state.rerun_log_paths['annotated_image_path']}/keypoints_{idx}",
            Points2DWithConfidence(
                positions=current_uv,
                confidences=scores.squeeze(),
                class_ids=2,
                keypoint_ids=COCO_17_IDS,
                colors=colors,
                radii=state.radii,
            ),
        )


@dataclass
class InputComponents:
    input_img: gr.Image
    point_type: gr.Radio
    annot_mode: gr.Radio
    current_state: gr.State
    sv_prediction: gr.State

    def to_list(self) -> list:
        return [getattr(self, f.name) for f in fields(self)]


@dataclass
class InputValues:
    img: np.ndarray
    point_type: Literal["include", "exclude"]
    annot_mode: Literal["Box", "Keypoint", "Segmentation"]
    current_state: CurrentState
    sv_prediction: SVPrediction


@dataclass
class VisibilityComponents:
    visbility_mode: gr.Radio
    current_state: gr.State
    sv_prediction: gr.State

    def to_list(self) -> list:
        return [getattr(self, f.name) for f in fields(self)]


@dataclass
class VisibilityValues:
    visbility_mode: Literal["Visible", "Occluded", "Out of Frame"]
    current_state: CurrentState
    sv_prediction: SVPrediction


@dataclass
class StateAndPredComponents:
    current_state: gr.State
    sv_prediction: gr.State

    def to_list(self) -> list:
        return [getattr(self, f.name) for f in fields(self)]


@dataclass
class StateAndPredValues:
    current_state: CurrentState
    sv_prediction: SVPrediction


def create_blueprint(state: CurrentState) -> rrb.Blueprint:
    blueprint = rrb.Blueprint(
        rrb.Horizontal(
            rrb.Vertical(
                rrb.Tabs(
                    rrb.Spatial2DView(
                        origin=f"{state.rerun_log_paths['original_image_path']}",
                        name="Original Image",
                    ),
                    rrb.Spatial2DView(
                        origin=f"{state.rerun_log_paths['pinhole_log_path']}/depth",
                        name="Depth Prediction",
                    ),
                    rrb.TextDocumentView(name="Info", origin="info"),
                ),
                rrb.Tabs(
                    rrb.Spatial2DView(
                        origin=f"{state.rerun_log_paths['annotated_image_path']}",
                        contents=[
                            "$origin/**",
                            f"- {state.rerun_log_paths['annotated_image_path']}/segmentation",
                        ],
                        name="Annotated Image",
                    ),
                    rrb.Spatial2DView(
                        origin=f"{state.rerun_log_paths['annotated_image_path']}",
                        name="Segmentation",
                    ),
                ),
            ),
            rrb.Spatial3DView(
                contents=[
                    "$origin/**",
                    f"- {state.rerun_log_paths['pinhole_log_path']}/depth",
                ]
            ),
            column_shares=[1, 1],
        ),
        collapse_panels=False,
    )
    return blueprint


# ----------------------------------------------------------------------
# Robust helper to turn the EncodedImage Blob (pyarrow.ListScalar) into
# a H×W×C NumPy array – works even when Arrow wraps the bytes in extra
# list levels (list<list<uint8>>).
# ----------------------------------------------------------------------


def _to_bytes(obj) -> bytes:
    """
    Recursively flattens a (possibly deeply‑nested) Python list of ints
    – as returned by `pyarrow.ListScalar.as_py()` – into a single
    `bytes` object that OpenCV/Pillow can consume.
    """
    if isinstance(obj, bytes | bytearray):
        return bytes(obj)
    if not isinstance(obj, list):
        # Single uint8 scalar
        return bytes([obj])

    flat: list[int] = []
    stack: list = [obj]
    while stack:  # iterative DFS – avoids recursion limit
        cur = stack.pop()
        if isinstance(cur, list):
            stack.extend(cur)
        else:
            flat.append(cur)
    flat.reverse()  # restore original order after stack pop
    return bytes(flat)


def decode_jpeg(blob_cell):
    """
    Parameters
    ----------
    blob_cell : pyarrow.ListScalar
        The `Blob` component of a Rerun EncodedImage (JPEG/PNG/...).

    Returns
    -------
    np.ndarray
        Decoded image array (H×W×3 or H×W×4 depending on channels).

    Notes
    -----
    Older Arrow versions sometimes store `Blob` as ``list<list<uint8>>``.
    This helper hides that quirk so the rest of the pipeline can treat it
    as a plain byte vector.
    """
    blob_bytes = _to_bytes(blob_cell.as_py())
    arr8 = np.frombuffer(blob_bytes, dtype=np.uint8)
    bgr = cv2.imdecode(arr8, cv2.IMREAD_UNCHANGED)
    if bgr is None:
        raise ValueError("cv2.imdecode failed – blob is not a valid image.")
    return bgr


# ----------------------------------------------------------------------
# Convert uncompressed Rerun ImageBuffer + ImageFormat to NumPy array
# ----------------------------------------------------------------------
def buffer_to_numpy(buf_cell, fmt_cell, dtype=np.uint8):
    """
    Convert an un‑compressed Rerun ImageBuffer + ImageFormat pair into a NumPy
    array.

    Parameters
    ----------
    buf_cell : pyarrow.ListScalar
        The ImageBuffer column cell (raw bytes as list<uint8> or list<list<uint8>>).
    fmt_cell : pyarrow.StructScalar
        The matching ImageFormat cell (contains width/height, etc.).
    dtype : np.dtype
        NumPy dtype of the underlying buffer (default uint8).

    Returns
    -------
    np.ndarray
        (H, W) for single‑channel images or (H, W, C) when multiple channels
        are present.
    """
    # Arrow sometimes stores ImageFormat as list<struct<...>> instead of a
    # plain struct scalar.  In that case we have to unwrap the single element.
    if hasattr(fmt_cell, "values"):  # ListScalar
        # expect exactly one struct inside
        fmt_struct = fmt_cell.values[0]
        fmt_dict = fmt_struct.as_py()
    else:  # StructScalar
        fmt_dict = fmt_cell.as_py()

    # after unwrapping we have the familiar dict with width/height
    h, w = fmt_dict["height"], fmt_dict["width"]

    data = _to_bytes(buf_cell.as_py())  # reuse the flattener from above
    arr = np.frombuffer(data, dtype=dtype)

    if arr.size == h * w:  # mono‑channel mask
        return arr.reshape(h, w)
    else:  # multi‑channel image
        channels = arr.size // (h * w)
        return arr.reshape(h, w, channels)


def load_annotation(state: CurrentState) -> SVPrediction:
    image_name: str = EXAMPLE_IMAGES[state.current_img_idx].stem

    json_sv_pred_path = rrd_dir / f"{image_name}.json"
    sv_prediction: SVPrediction = from_json(SVPrediction, json_sv_pred_path.read_text())

    rrd_path: Path = rrd_dir / f"{image_name}.rrd"
    assert rrd_path.exists(), f"Rerun recording {rrd_path} does not exist."
    saved_recording = rr.dataframe.load_recording(rrd_path)
    # query the recording into a pandas dataframe
    view = saved_recording.view(
        index="iteration",
        contents=str(state.rerun_log_paths["original_image_path"]),
    )
    table = view.select().read_all()
    # print("Table shape:", table)
    # rgb_hw3 = np.array(table["/world/camera/pinhole/original:Blob"][0])
    # # rgb_hw3 = decode_jpeg(table["/world/camera/pinhole/original:Blob"][0])
    # print(rgb_hw3.shape)
    # ❶ extract the scalar that holds the JPEG bytes
    blob_scalar = table["/world/camera/pinhole/original:Blob"][0]  # list<uint8> scalar
    bgr = decode_jpeg(blob_scalar)
    rgb_hw3 = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    annotated_view = saved_recording.view(
        index="iteration",
        contents=str(state.rerun_log_paths["annotated_image_path"] / "segmentation"),
    )

    segmentation = annotated_view.select().read_all()
    # Column names in the Arrow table always start with a leading slash.
    seg_base = f"/{state.rerun_log_paths['annotated_image_path']}/segmentation"
    seg_blob_scalar = segmentation[f"{seg_base}:ImageBuffer"][0]
    seg_fmt_scalar = segmentation[f"{seg_base}:ImageFormat"][0]
    seg_img = buffer_to_numpy(seg_blob_scalar, seg_fmt_scalar)

    # ------------------------------------------------------------------
    # 💧 Load the depth image that we logged as rr.DepthImage
    # ------------------------------------------------------------------
    depth_view = saved_recording.view(
        index="iteration",
        contents=str(state.rerun_log_paths["pinhole_log_path"] / "depth"),
    )
    depth_table = depth_view.select().read_all()

    depth_base = f"/{state.rerun_log_paths['pinhole_log_path']}/depth"
    try:
        depth_blob = depth_table[f"{depth_base}:ImageBuffer"][0]
        depth_fmt = depth_table[f"{depth_base}:ImageFormat"][0]
    except KeyError:
        # Older Rerun versions store DepthImage directly in a single column
        # called 'DepthImage'.  Fall back to that if present.
        depth_blob = depth_table[f"{depth_base}:DepthImage"][0]
        depth_fmt = None  # No explicit format; we'll assume float32 rows×cols

    # Depth is stored as 32‑bit float instead of uint8
    depth_img = buffer_to_numpy(depth_blob, depth_fmt, dtype=np.float32) if depth_fmt else np.asarray(depth_blob)

    # Populate the prediction object so the rest of the UI can access it
    sv_prediction.depth_relative = depth_img

    sv_prediction.rgb_hw3 = rgb_hw3
    sv_prediction.mask = seg_img.astype(bool)
    sv_prediction.depth_relative = sv_prediction.depth_relative.astype(np.float32)

    return sv_prediction


def annotate_image(*input_params):
    input_params = InputValues(*input_params)
    if input_params.annot_mode == "Segmentation":
        raise gr.Error("Keypoint Segmentation mode is not implemented yet.")

    current_state: CurrentState = input_params.current_state
    sv_prediction: SVPrediction = input_params.sv_prediction

    # create a new recording with a unique id
    rec = get_recording(current_state.recording_id)
    stream = rec.binary_stream()

    # check if the current image is already annotated
    rec.set_time("iteration", sequence=0)
    rec.log("/", rr.ViewCoordinates.RDF, static=True)

    image_name = EXAMPLE_IMAGES[current_state.current_img_idx].stem
    if image_name in LABEL_INFO.annotated:
        sv_prediction: SVPrediction = load_annotation(current_state)
        log_sv_pred(rec, current_state, sv_prediction)
    else:
        blueprint = create_blueprint(current_state)

        new_annotation_context(rec)

        rec.send_blueprint(blueprint)

        info_str = f"Image Name: {image_name}"
        sv_prediction.info_message = info_str

        rec.log(
            "info",
            rr.TextDocument(
                text=info_str,
                media_type="markdown",
            ),
        )

        rgb_hw3: UInt8[ndarray, "H W 3"] = input_params.img
        # resize image so that the maximum dimension is 1024 pixels
        max_dim: int = 1024
        if max(rgb_hw3.shape[0], rgb_hw3.shape[1]) > max_dim:
            scale_factor: float = max_dim / max(rgb_hw3.shape[0], rgb_hw3.shape[1])
            new_size: tuple[int, int] = (int(rgb_hw3.shape[1] * scale_factor), int(rgb_hw3.shape[0] * scale_factor))
            rgb_hw3 = np.array(Image.fromarray(rgb_hw3).resize(new_size, Image.LANCZOS))

        # convert to pil image
        rgb_pil: Image.Image = Image.fromarray(rgb_hw3, mode="RGB")
        segmented_rgba: Image.Image = remove_bg(rgb_pil)

        # Convert PIL RGBA image to NumPy array
        rgba_data_np: UInt8[ndarray, "H W 4"] = np.array(segmented_rgba)

        # Initialize segmented_rgb with the RGB channels.
        # The type hint reflects the actual shape (H, W, 3).
        segmented_rgb: UInt8[ndarray, "H W 3"] = rgba_data_np[:, :, :3].copy()

        # Create a mask for pixels where the alpha channel is 0.
        # This identifies transparent background areas from the segmentation.
        transparent_mask: Bool[ndarray, "H W"] = rgba_data_np[:, :, 3] == 0
        binary_segmentation_mask = transparent_mask < 0.5

        min_dim = min(rgb_hw3.shape[0], rgb_hw3.shape[1])
        # Heuristic for radii based on image size. Adjust the divisor as needed.
        radii_value = max(1.0, min_dim / 100.0)

        # Create a bounding box from the binary_segmentation_mask, only do so if theres only one bbox and its topleft/right are empty
        xyxy_list: list[XYXYContainer] = sv_prediction.xyxy_list
        if len(xyxy_list) == 1 and (xyxy_list[0].top_left.shape[0] == 0 and xyxy_list[0].bottom_right.shape[0] == 0):
            rows = np.any(binary_segmentation_mask, axis=1)
            cols = np.any(binary_segmentation_mask, axis=0)
            if rows.any() and cols.any():
                ymin, ymax = np.where(rows)[0][[0, -1]]
                xmin, xmax = np.where(cols)[0][[0, -1]]
                bbox_xyxy = np.array([xmin, ymin, xmax, ymax])
                # update xyxy_list with the new bounding box
                xyxy_list[0].add_point((float(xmin), float(ymin)), "top_left")
                xyxy_list[0].add_point((float(xmax), float(ymax)), "bottom_right")
                # Log the bounding box
                rec.log(
                    f"{current_state.rerun_log_paths['annotated_image_path']}/bbox_{current_state.current_xyxy_idx}",
                    rr.Boxes2D(
                        array=bbox_xyxy,
                        array_format=rr.Box2DFormat.XYXY,
                        labels=[f"Person {current_state.current_xyxy_idx}"],
                        colors=(255, 0, 0),
                        show_labels=True,
                    ),
                )

                rec.log(
                    f"{current_state.rerun_log_paths['annotated_image_path']}/top_left_{current_state.current_xyxy_idx}",
                    rr.Points2D(bbox_xyxy[0:2], colors=(0, 0, 255), radii=radii_value, show_labels=True),
                )

                # Log exclude points if any exist
                rec.log(
                    f"{current_state.rerun_log_paths['annotated_image_path']}/bottom_right{current_state.current_xyxy_idx}",
                    rr.Points2D(bbox_xyxy[2:4], colors=(255, 0, 0), radii=radii_value),
                )
            else:
                # Handle case where mask is empty or all False
                bbox_xyxy = None  # Or some default like np.array([0,0,0,0])
        else:
            bbox_mask = np.zeros(binary_segmentation_mask.shape, dtype=bool)
            H, W = bbox_mask.shape
            for i, xyxy in enumerate(xyxy_list):
                if xyxy.bbox is not None and np.array(xyxy.bbox).size == 4:
                    flat_bbox = np.array(xyxy.bbox).flatten()
                    xmin, ymin, xmax, ymax = map(int, flat_bbox)
                    xmin = max(0, min(xmin, xmax, W - 1))
                    xmax = max(0, min(max(xmin, xmax), W))
                    ymin = max(0, min(ymin, ymax, H - 1))
                    ymax = max(0, min(max(ymin, ymax), H))
                    if xmax > xmin and ymax > ymin:
                        bbox_mask[ymin:ymax, xmin:xmax] = True

            binary_segmentation_mask = binary_segmentation_mask & bbox_mask
            transparent_mask = ~binary_segmentation_mask

        # Set RGB values to black [0, 0, 0] for these transparent areas.
        # This is done before passing to the depth predictor.
        segmented_rgb[transparent_mask] = [0, 0, 0]

        rec.log(
            f"{current_state.rerun_log_paths['original_image_path']}", rr.Image(rgb_hw3, rr.ColorModel.RGB).compress()
        )
        rec.log(
            f"{current_state.rerun_log_paths['annotated_image_path']}",
            rr.Image(segmented_rgb, rr.ColorModel.RGB).compress(),
        )
        rec.log(
            f"{current_state.rerun_log_paths['annotated_image_path']}/segmentation",
            rr.SegmentationImage(binary_segmentation_mask.astype(np.uint8)),
        )

        relative_pred: RelativeDepthPrediction = depth_predictor.__call__(rgb=segmented_rgb, K_33=None)
        depth_hw: Float32[np.ndarray, "H W"] = log_relative_pred(
            rec, current_state.rerun_log_paths["parent_log_path"], relative_pred, rgb_hw3, depth_edge_threshold=0.1
        )

        intri = Intrinsics(
            camera_conventions="RDF",
            fl_x=float(relative_pred.K_33[0, 0]),
            fl_y=float(relative_pred.K_33[1, 1]),
            cx=float(relative_pred.K_33[0, 2]),
            cy=float(relative_pred.K_33[1, 2]),
            width=rgb_hw3.shape[1],
            height=rgb_hw3.shape[0],
        )
        extri = Extrinsics(
            world_R_cam=np.eye(3, dtype=np.float64),
            world_t_cam=np.zeros(3, dtype=np.float64),
        )
        pinhole_params = PinholeParameters(name="camera", extrinsics=extri, intrinsics=intri)
        sv_prediction.pinhole_params = pinhole_params

        uvc_list = sv_prediction.uvc_list

        match input_params.annot_mode:
            case "Box":
                pass
            case "Keypoint":
                # first check if there are already keypoints in the sv_prediction
                if len(uvc_list) == 0:
                    bboxes: Float[ndarray, "n_dets 4"] = np.concat(
                        [xyxy.bbox for xyxy in sv_prediction.xyxy_list if xyxy.bbox is not None]
                    )
                    model_outputs = kpt_only_model(rgb_hw3, bboxes=bboxes)
                    uv_preds: Float[ndarray, "n_dets n_kpts 2"] = model_outputs[0][:, 0:17, :]
                    # scores: Float[ndarray, "n_dets n_kpts"] = model_outputs[1][:, 0:17]

                    # set all scores to 1.0
                    scores: Float[ndarray, "n_dets n_kpts"] = np.ones(
                        (uv_preds.shape[0], uv_preds.shape[1]), dtype=np.float32
                    )
                    confidence_colors: UInt8[ndarray, "n_dets n_kpts 3"] = np.full(
                        (uv_preds.shape[0], uv_preds.shape[1], 3), [0, 255, 0], dtype=np.uint8
                    )

                    for i, (uv, score, color) in enumerate(zip(uv_preds, scores, confidence_colors, strict=True)):
                        # convert from coco133 to coco17
                        uv_17: Float[ndarray, "17 2"] = uv.copy()[0:17, :]
                        score17: Float[ndarray, "17"] = score.copy()[0:17]
                        # Only set for the currently selected bounding box
                        if i == current_state.current_xyxy_idx:
                            color[current_state.current_uvc_idx] = SELECTED_COLOR
                        rec.log(
                            f"{current_state.rerun_log_paths['annotated_image_path']}/keypoints_{i}",
                            Points2DWithConfidence(
                                positions=uv_17,
                                confidences=score17,
                                class_ids=2,
                                keypoint_ids=COCO_17_IDS,
                                colors=color,
                                radii=radii_value,
                            ),
                        )
                        uvc: Float[ndarray, "n_kpts 3"] = np.concatenate([uv_17, score17[:, np.newaxis]], axis=-1)
                        # update kpts_list
                        uvc_list.append(uvc)
            case "Segmentation":
                ...
                # keypoints for segmentation, only use eyes + shoulders + hips
                # all_xyxy: Float[ndarray, "n_dets 4"] = np.concat(
                #     [xyxy.bbox for xyxy in input_params.xyxy_list if xyxy.bbox is not None]
                # )
                # for i, (xy, xyxy) in enumerate(zip(xyc_list, all_xyxy, strict=True)):
                #     # convert from coco133 to coco17
                #     xy_17: Float[ndarray, "17 2"] = xy.copy()[0:17, :]
                #     seg_kpts_ids: list[int] = [
                #         COCO_17_NAME2ID[name]
                #         for name in ["left_eye", "right_eye", "left_shoulder", "right_shoulder", "left_hip", "right_hip"]
                #     ]
                #     include_points: Float[ndarray, "6 2"] = xy_17[[seg_kpts_ids], :].squeeze()
                #     labels: Int[np.ndarray, "num_points"] = np.ones(len(include_points), dtype=np.int32)  # noqa: F821

                #     with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                #         predictor.set_image(rgb_hw3)
                #         # mask_input : np.ndarray
                #         # A low resolution mask input to the model, typically coming from a previous prediction iteration. Has form 1xHxW, where for SAM, H=W=256.
                #         mask_input = None

                #         masks, _, _ = predictor.predict(
                #             point_coords=include_points,
                #             point_labels=labels,
                #             box=xyxy,
                #             mask_input=mask_input,
                #             multimask_output=False,
                #         )
                #         # masks, _, _ = predictor.predict(box=bbox, multimask_output=False)
                #         print(masks.shape)

                #     rec.log(
                #         f"{input_params.rerun_log_paths['annotated_image_path']}/segmentation2",
                #         rr.SegmentationImage(image=masks[0]),
                #     )

                #     new_segmented_rgb: UInt8[ndarray, "H W 3"] = rgb_hw3.copy()
                #     # Apply the mask to the RGB image
                #     # Convert the PyTorch tensor mask to a boolean NumPy array for indexing
                #     mask_for_indexing: Bool[np.ndarray, "H W"] = masks[0].astype(bool)
                #     new_segmented_rgb[~mask_for_indexing] = [0, 0, 0]

                #     relative_pred: RelativeDepthPrediction = depth_predictor.__call__(rgb=new_segmented_rgb, K_33=None)
                #     log_relative_pred(
                #         rec,
                #         input_params.rerun_log_paths["parent_log_path"],
                #         relative_pred,
                #         rgb_hw3,
                #         depth_edge_threshold=0.1,
                #     )

        # update sv_prediction with the new rgb_hw3, mask, depth_relative
        sv_prediction.rgb_hw3 = rgb_hw3
        sv_prediction.mask = binary_segmentation_mask
        sv_prediction.depth_relative = depth_hw

    yield stream.read(), sv_prediction, current_state


def build_annot_ui():
    with gr.Row():
        input_img = gr.Image(type="numpy", visible=False)

    INITIAL_INDEX = -1  # Indicates no image is selected yet

    current_state = gr.State(
        CurrentState(
            recording_id=uuid.uuid4(),
            rerun_log_paths=RerunLogPaths(
                timeline_name="iteration",
                parent_log_path=Path("world"),
                camera_log_path=Path("world/camera"),
                pinhole_log_path=Path("world/camera/pinhole"),
                original_image_path=Path("world/camera/pinhole/original"),
                annotated_image_path=Path("world/camera/pinhole/annotated"),
            ),
            current_img_idx=INITIAL_INDEX,  # Initialize to -1 to indicate no image is selected yet
            current_xyxy_idx=0,
            current_uvc_idx=0,
            radii=5.0,  # Default radius for keypoints
        )
    )

    sv_prediction = gr.State(
        SVPrediction(
            seg_kpts=SAM2KeypointContainer.empty(),
            uvc_list=[],
            xyxy_list=[XYXYContainer.empty()],
            info_message="",
            rgb_hw3=None,  # Will be set during processing
            mask=None,  # Will be set during processing
            depth_relative=None,  # Will be set during processing
            pinhole_params=None,  # Will be set during processing
        )
    )

    status: str = (
        f"Image: **{current_state.value.current_img_idx}/{len(EXAMPLE_IMAGES)}**\n\n"
        f"Current image: **{EXAMPLE_IMAGES[current_state.value.current_img_idx].stem}**\n\n"
    )

    with gr.Row():
        with gr.Tab(label="Annotation UI", scale=2), gr.Column():
            with gr.Row() as start_ui, gr.Column():
                start_btn = gr.Button("Start Labeling", visible=True)
                with gr.Row():
                    random_order_switch = gr.Checkbox(
                        label="Random Order",
                        value=False,
                        visible=True,
                    )
                    show_labeled_switch = gr.Checkbox(
                        label="Show Labeled Images",
                        value=True,
                        visible=True,
                    )
            with gr.Row():
                prev_img_btn = gr.Button("Previous Image", visible=False)
                next_img_btn = gr.Button("Next Image", visible=False)
            random_img_btn = gr.Button("Random Image", visible=False)
            skip_btn = gr.Button("Skip Image", visible=False)
            status_md = gr.Markdown(
                value=status,
                show_label=False,
                container=False,
            )
            save_btn = gr.Button("Save Annotations", visible=False)
            annot_mode = gr.Radio(
                label="Annotation Mode",
                choices=["Box", "Keypoint", "Segmentation"],
                value="Box",
                visible=True,
            )
            ### Bounding Box Annotation UI Controls ###
            with gr.Column(visible=True) as bbox_controls:  # Group the bounding box controls
                bbox_type = gr.Radio(
                    label="bbox type", choices=["top_left", "bottom_right"], value="top_left", visible=True
                )
                with gr.Row():
                    add_bbox_btn = gr.Button("Add BBox")
                    remove_bbox_btn = gr.Button("Remove BBox")
            with gr.Group():
                gr.Markdown(
                    value="Choose Bounding Box with < >",
                    show_label=False,
                    container=False,
                )
                with gr.Row(equal_height=True):
                    box_btn_left = gr.Button("‹")
                    box_btn_right = gr.Button("›")
                with gr.Row():
                    current_bbox_text = gr.Markdown(
                        value=f"Current Person: **{current_state.value.current_xyxy_idx}**",
                        show_label=False,
                        container=False,
                    )
            ### Bounding Box Annotation UI Controls ###

            ### Keypoint Annotation UI Controls ###
            with gr.Column(visible=False) as keypoint_controls:  # Group the keypoint controls
                with gr.Row(equal_height=True):
                    kpt_btn_left = gr.Button("‹")
                    kpt_btn_right = gr.Button("›")
                current_kpts_name = gr.Markdown(
                    value=f"Current Keypoint: **{list(COCO_17_NAME2ID.keys())[0]}**",
                    show_label=False,
                    container=False,
                )

                kpt_visibility = gr.Radio(
                    label="Keypoint Visibility",
                    choices=["Visible", "Occluded", "Out of Frame"],
                    value="Visible",
                )
            ### Keypoint Annotation UI Controls ###
            ### Segmentation Annotation UI Controls ###
            with gr.Column(visible=False) as segmentation_controls:  # Group the keypoint controls
                point_type = gr.Radio(label="point type", choices=["include", "exclude"], value="include")
                ### Segmentation Annotation UI Controls ###
        with gr.Tab(label="Skip Json", scale=2):
            json_output = gr.JSON(value=to_json(LABEL_INFO))
        with gr.Column(scale=5):
            viewer = Rerun(
                streaming=True,
                panel_states={
                    "time": "collapsed",
                    "blueprint": "collapsed",
                    "selection": "hidden",
                },
                height=850,
            )

    viewer.selection_change(
        single_view_update_keypoints,
        inputs=[
            annot_mode,
            point_type,
            bbox_type,
            current_state,
            sv_prediction,
        ],
        outputs=[viewer],
    )

    input_components = InputComponents(
        input_img=input_img,
        point_type=point_type,
        annot_mode=annot_mode,
        current_state=current_state,
        sv_prediction=sv_prediction,
    )

    def update_state(idx: int):
        new_status: str = f"Image: **{idx}/{len(EXAMPLE_IMAGES)}**\n\nCurrent image: **{EXAMPLE_IMAGES[idx].stem}**\n\n"

        new_state = CurrentState(
            recording_id=uuid.uuid4(),
            rerun_log_paths=RerunLogPaths(
                timeline_name="iteration",
                parent_log_path=Path("world"),
                camera_log_path=Path("world/camera"),
                pinhole_log_path=Path("world/camera/pinhole"),
                original_image_path=Path("world/camera/pinhole/original"),
                annotated_image_path=Path("world/camera/pinhole/annotated"),
            ),
            current_img_idx=idx,
            current_xyxy_idx=0,  # Reset to first bbox
            current_uvc_idx=0,  # Reset to first keypoint
            radii=5.0,  # Default radius for keypoints
        )
        sv_prediction = SVPrediction(
            seg_kpts=SAM2KeypointContainer.empty(),
            uvc_list=[],
            xyxy_list=[XYXYContainer.empty()],  # Reset to empty bounding box list
            rgb_hw3=None,  # Will be set during processing
            mask=None,  # Will be set during processing
            depth_relative=None,  # Will be set during processing
            pinhole_params=None,  # Will be set during processing
            info_message="",  # Reset info message
        )
        # Always return a new list and new container
        return (
            "Box",  # Reset to Box mode
            gr.Markdown(value=new_status, show_label=False, container=False),
            new_state,
            sv_prediction,
        )

    class Action(Enum):
        NEXT = auto()
        PREV = auto()
        RAND = auto()
        SKIP = auto()

    def load_image(action: Action, current_state: CurrentState, show_labeled: bool = True) -> tuple:
        """Navigate/skip images and return the huge Gradio tuple."""
        if not EXAMPLE_IMAGES:  # nothing left at all
            return None, 0, uuid.uuid4(), "", *update_state(0)

        json_str: str = to_json(LABEL_INFO)
        current_idx = current_state.current_img_idx

        # ------------------------------------------------------------------
        # Handle side-effects unique to SKIP **before** we decide new_idx
        # ------------------------------------------------------------------
        if action is Action.SKIP:
            LABEL_INFO.skip.append(EXAMPLE_IMAGES[current_idx].stem)
            json_str: str = to_json(LABEL_INFO)
            label_info_json_path.write_text(json_str)
            EXAMPLE_IMAGES.pop(current_idx)
            if not EXAMPLE_IMAGES:  # skipped the last one
                return None, 0, uuid.uuid4(), json_str, *update_state(0)

        # ------------------------------------------------------------------
        # Structural pattern-matching chooses the next index
        # ------------------------------------------------------------------
        match action:
            case Action.NEXT:
                new_idx = (current_idx + 1) % len(EXAMPLE_IMAGES)
            case Action.PREV:
                new_idx = (current_idx - 1 + len(EXAMPLE_IMAGES)) % len(EXAMPLE_IMAGES)
            case Action.RAND:
                new_idx = random.randrange(len(EXAMPLE_IMAGES))
            case Action.SKIP:  # treat like NEXT after pop
                new_idx = current_idx % len(EXAMPLE_IMAGES)
            case _:  # theoretically unreachable
                raise ValueError(f"Unsupported action: {action}")

        # Skip over any images already flagged as skipped
        while EXAMPLE_IMAGES[new_idx].stem in LABEL_INFO.skip:
            new_idx = (new_idx + 1) % len(EXAMPLE_IMAGES)

        # Skip over labeled images if show_labeled is False
        if not show_labeled:
            start_idx = new_idx
            while (
                EXAMPLE_IMAGES[new_idx].stem in LABEL_INFO.annotated or EXAMPLE_IMAGES[new_idx].stem in LABEL_INFO.skip
            ):
                new_idx: int = (new_idx + 1) % len(EXAMPLE_IMAGES)
                # Safety check to avoid infinite loop if all images are labeled
                if new_idx == start_idx:
                    break  # Break after a full cycle to avoid infinite loop

        img_path = str(EXAMPLE_IMAGES[new_idx])
        return (img_path, json_str, *update_state(new_idx))

    next_img_btn.click(
        fn=lambda state, show_labeled: load_image(Action.NEXT, state, show_labeled),
        inputs=[current_state, show_labeled_switch],
        outputs=[
            input_img,
            json_output,
            annot_mode,
            status_md,
            current_state,
            sv_prediction,
        ],
    )

    prev_img_btn.click(
        fn=lambda state, show_labeled: load_image(Action.PREV, state, show_labeled),
        inputs=[current_state, show_labeled_switch],
        outputs=[
            input_img,
            json_output,
            annot_mode,
            status_md,
            current_state,
            sv_prediction,
        ],
    )

    random_img_btn.click(
        fn=lambda state, show_labeled: load_image(Action.RAND, state, show_labeled),
        inputs=[current_state, show_labeled_switch],
        outputs=[
            input_img,
            json_output,
            annot_mode,
            status_md,
            current_state,
            sv_prediction,
        ],
    )

    skip_btn.click(
        fn=lambda state, show_labeled: load_image(Action.SKIP, state, show_labeled),
        inputs=[current_state, show_labeled_switch],
        outputs=[
            input_img,
            json_output,
            annot_mode,
            status_md,
            current_state,
            sv_prediction,
        ],
    )

    def start_fn(state: CurrentState, random_order: bool):
        """Initialize the annotation UI by loading the first image and showing controls."""
        current_img_idx = state.current_img_idx
        if random_order:
            current_img_idx = random.randint(0, len(EXAMPLE_IMAGES) - 1)

        state.current_img_idx = current_img_idx
        return (
            # str(image_path),
            gr.update(visible=False),  # Hide start_btn
            gr.Button(visible=True),
            gr.Button(visible=True),  # Show prev_img_btn
            gr.Button(visible=True),  # Show random_img_btn
            gr.Button(visible=True),  # Show skip_btn
            gr.Button(visible=True),  # Show save_btn
            state,  # Initialize current_image_index
        )

    # Modify start_btn to initialize current_image_index and load the first image
    start_btn.click(
        fn=start_fn,
        inputs=[current_state, random_order_switch],
        outputs=[
            start_ui,
            next_img_btn,
            prev_img_btn,
            random_img_btn,
            skip_btn,
            save_btn,
            current_state,
        ],
    ).then(
        fn=lambda state, show_labeled: load_image(Action.NEXT, state, show_labeled),
        inputs=[current_state, show_labeled_switch],
        outputs=[
            input_img,
            json_output,
            annot_mode,
            status_md,
            current_state,
            sv_prediction,
        ],
    )

    def save_annot_fn(state: CurrentState, sv_prediction: SVPrediction) -> str:
        """Save the current annotations to the label_info file."""

        rec = get_recording(state.recording_id)
        stream = rec.binary_stream()
        name: str = EXAMPLE_IMAGES[state.current_img_idx].stem

        log_sv_pred(rec, state, sv_prediction)

        if name not in LABEL_INFO.annotated:
            LABEL_INFO.annotated.append(name)
        # Update the annotation info file
        json_str: str = to_json(LABEL_INFO)
        label_info_json_path.write_text(json_str)

        # make sure that all recorded data has been encoded into the stream
        rrd_save_path: Path = rrd_dir / f"{name}.rrd"
        # save the current recording
        with rrd_save_path.open("wb") as f:
            f.write(stream.read(flush=True))
        # save sv_prediction to json
        json_save_path: Path = rrd_save_path.with_suffix(".json")
        sv_pred_dict = to_json(sv_prediction)
        json_save_path.write_text(sv_pred_dict)

        print(f"Saved recording to {rrd_save_path}")

        return json_str

    save_btn.click(
        fn=save_annot_fn,
        inputs=[current_state, sv_prediction],
        outputs=[json_output],
    ).then(
        fn=lambda state: load_image(Action.NEXT, state),
        inputs=[current_state],
        outputs=[
            input_img,
            json_output,
            annot_mode,
            status_md,
            current_state,
            sv_prediction,
        ],
    )

    ## Annotation Mode Toggle ##
    def toggle_annotation_mode(mode: str):
        if mode == "Box":
            return gr.update(visible=True), gr.update(visible=False), gr.update(visible=False)
        elif mode == "Keypoint":
            return gr.update(visible=False), gr.update(visible=True), gr.update(visible=False)
        elif mode == "Segmentation":
            return gr.update(visible=False), gr.update(visible=False), gr.update(visible=True)

    annot_mode.change(
        fn=toggle_annotation_mode,
        inputs=[annot_mode],
        outputs=[bbox_controls, keypoint_controls, segmentation_controls],
    ).then(
        fn=annotate_image,
        inputs=input_components.to_list(),
        outputs=[viewer, sv_prediction, current_state],
    )

    ## Annotation Mode Toggle ##
    def set_kpt_visbility(*vis_params):
        vis_values = VisibilityValues(*vis_params)

        state: CurrentState = vis_values.current_state
        sv_prediction: SVPrediction = vis_values.sv_prediction
        rec = get_recording(state.recording_id)
        stream = rec.binary_stream()

        current_uvc: Float[ndarray, "n_kpts 3"] = sv_prediction.uvc_list[state.current_xyxy_idx]
        current_uv: Float[ndarray, "n_kpts 2"] = current_uvc[:, 0:2]  # Extract x, y coordinates
        scores: Float[ndarray, "n_kpts"] = current_uvc[:, -1].astype(np.float32)  # Placeholder for scores
        # set scores based on visibility mode
        match vis_values.visbility_mode:
            case "Visible":
                scores[state.current_uvc_idx] = 1.0
            case "Occluded":
                scores[state.current_uvc_idx] = 0.5
            case "Out of Frame":
                scores[state.current_uvc_idx] = 0.0
                # also set current_xy to 0,0
                current_uv[state.current_uvc_idx] = [np.nan, np.nan]
        colors: UInt8[np.ndarray, "n_kpts 3"] = confidence_scores_to_rgb(
            confidence_scores=scores[np.newaxis, :, np.newaxis]
        ).squeeze()

        rec.log(
            f"{state.rerun_log_paths['annotated_image_path']}/keypoints_{state.current_xyxy_idx}",
            Points2DWithConfidence(
                positions=current_uv,
                confidences=scores.squeeze(),
                class_ids=2,
                keypoint_ids=COCO_17_IDS,
                colors=colors,
                radii=state.radii,
            ),
        )
        # update kpts_list
        sv_prediction.uvc_list[state.current_xyxy_idx] = np.concatenate([current_uv, scores[:, np.newaxis]], axis=-1)
        yield stream.read(), sv_prediction

    vis_components = VisibilityComponents(
        visbility_mode=kpt_visibility,
        current_state=current_state,
        sv_prediction=sv_prediction,
    )

    kpt_visibility.change(
        fn=set_kpt_visbility,
        inputs=vis_components.to_list(),
        outputs=[viewer, sv_prediction],
    )

    # needed because of beartype
    def add_bbox_fn(
        rgb_hw3,
        state,
        sv_prediction,
    ):
        yield from _add_bbox_fn(rgb_hw3, state, sv_prediction)  # <-- magic happens

    def _add_bbox_fn(
        rgb_hw3: UInt8[ndarray, "H W 3"],
        state: CurrentState,
        sv_prediction: SVPrediction,
    ):
        """
        Accept the current bounding box and update the list and index.
        """

        # get the recording
        rec: rr.RecordingStream = get_recording(state.recording_id)
        stream: rr.BinaryStream = rec.binary_stream()

        max_dim: int = 1024
        if max(rgb_hw3.shape[0], rgb_hw3.shape[1]) > max_dim:
            scale_factor: float = max_dim / max(rgb_hw3.shape[0], rgb_hw3.shape[1])
            new_size: tuple[int, int] = (int(rgb_hw3.shape[1] * scale_factor), int(rgb_hw3.shape[0] * scale_factor))
            rgb_hw3 = np.array(Image.fromarray(rgb_hw3).resize(new_size, Image.LANCZOS))

        img_h, img_w, _ = rgb_hw3.shape
        # create a bounding box with the center of the image, that has heigh width of 1/8 the original image
        cx, cy = img_w / 2, img_h / 2
        box_w, box_h = img_w // 8, img_h // 8
        top_left = (cx - box_w / 2, cy - box_h / 2)
        bottom_right = (cx + box_w / 2, cy + box_h / 2)
        xyxy_container: XYXYContainer = XYXYContainer.empty()
        xyxy_container.add_point(top_left, "top_left")
        xyxy_container.add_point(bottom_right, "bottom_right")

        sv_prediction.xyxy_list.append(xyxy_container)

        state.current_xyxy_idx += 1
        total_bbox_text: str = f"Current Person: **{state.current_xyxy_idx}**"

        xyxy: XYXYContainer
        for idx, xyxy in enumerate(sv_prediction.xyxy_list):
            # update visualization for all bounding boxes
            color = (255, 0, 0) if state.current_xyxy_idx == idx else SELECTED_XYXY_COLOR

            rec.log(
                f"{state.rerun_log_paths['annotated_image_path']}/bbox_{idx}",
                rr.Boxes2D(
                    array=xyxy.bbox,
                    array_format=rr.Box2DFormat.XYXY,
                    labels=[f"Person {idx}"],
                    colors=color,
                    show_labels=True,
                ),
            )

        yield (
            stream.read(),
            "top_left",  # Reset bbox type to top_left
            total_bbox_text,
            state,
            sv_prediction,
        )

    add_bbox_btn.click(
        fn=add_bbox_fn,
        inputs=[input_img, current_state, sv_prediction],
        outputs=[
            viewer,
            bbox_type,
            current_bbox_text,
            current_state,
            sv_prediction,
        ],
    )

    def remove_bbox_fn(
        state,
        sv_prediction,
    ):
        yield from _remove_bbox_fn(state, sv_prediction)  # <-- magic happens

    def _remove_bbox_fn(
        state: CurrentState,
        sv_prediction: SVPrediction,
    ):
        """
        Accept the current bounding box and update the list and index.
        """

        # get the recording
        rec: rr.RecordingStream = get_recording(state.recording_id)
        stream: rr.BinaryStream = rec.binary_stream()

        num_xyxy: int = len(sv_prediction.xyxy_list)
        if num_xyxy == 1:
            # don't do anything if there is only one bounding box, there has to be at least one bounding box
            raise gr.Error("Cannot remove the last bounding box. There needs to be at least one bounding box present.")
        elif num_xyxy > 1:
            # Directly remove the current bounding box using the known index
            idx = state.current_xyxy_idx
            sv_prediction.xyxy_list.pop(idx)
            # Remove the current keypoints for this bounding box
            sv_prediction.uvc_list.pop(idx)
            # Reset the current bounding box index to 0
            state.current_xyxy_idx = 0
            # Clear the xyxy from rerun
            rec.log(f"{state.rerun_log_paths['annotated_image_path']}/bbox_{idx}", rr.Clear(recursive=True))
            # Clear the keypoints for this bounding box
            rec.log(f"{state.rerun_log_paths['annotated_image_path']}/keypoints_{idx}", rr.Clear(recursive=True))

        # state.current_xyxy_idx += 1
        total_bbox_text: str = f"Current Person: **{state.current_xyxy_idx}**"

        yield (
            stream.read(),
            "top_left",  # Reset bbox type to bottom_right
            total_bbox_text,
            state,
            sv_prediction,
        )

    def navigate_bbox(
        direction: int, state: CurrentState, sv_pred: SVPrediction
    ) -> tuple[str, str, CurrentState, SVPrediction]:
        """
        Navigate through the bounding boxes.
        direction: -1 for left, +1 for right.
        current_idx: The current index of the bounding box.
        current_xyxy_list: The list of bounding boxes.
        Returns: (new_index, new_markdown_text_for_current_person)
        """
        list_len: int = len(sv_pred.xyxy_list)

        if list_len == 0:
            # This case should ideally not be hit if xyxy_list is initialized with at least one item.
            new_idx = 0
        elif direction == -1:  # Move left
            new_idx = max(0, state.current_xyxy_idx - 1)
        elif direction == +1:  # Move right
            # current_idx can go from 0 to list_len - 1.
            # If current_idx is list_len - 1 (last item), it should stay list_len - 1.
            new_idx = min(list_len - 1, state.current_xyxy_idx + 1)
        else:
            # Should not happen with direction being -1 or +1
            new_idx = state.current_xyxy_idx

        # reset the chosen keypoint index to 0 when navigating bounding boxes
        state.current_xyxy_idx = new_idx
        state.current_uvc_idx = 0
        current_kpt_str: str = list(COCO_17_NAME2ID.keys())[state.current_uvc_idx]
        return f"Current Person: **{new_idx}**", f"Current Keypoint: **{current_kpt_str}**", state, sv_pred

    def update_bbox_color(*state_and_pred_params):
        state_and_pred = StateAndPredValues(*state_and_pred_params)
        state: CurrentState = state_and_pred.current_state
        sv_prediction: SVPrediction = state_and_pred.sv_prediction

        rec: rr.RecordingStream = get_recording(state.recording_id)
        stream: rr.BinaryStream = rec.binary_stream()

        xyxy: XYXYContainer
        for idx, xyxy in enumerate(sv_prediction.xyxy_list):
            # update visualization for all bounding boxes
            color = (255, 0, 0) if state.current_xyxy_idx == idx else SELECTED_XYXY_COLOR

            rec.log(
                f"{state.rerun_log_paths['annotated_image_path']}/bbox_{idx}",
                rr.Boxes2D(
                    array=xyxy.bbox,
                    array_format=rr.Box2DFormat.XYXY,
                    labels=[f"Person {idx}"],
                    colors=color,
                    show_labels=True,
                ),
            )
        yield stream.read()

    def update_selected_keypoint(*state_and_pred_params):
        state_and_pred = StateAndPredValues(*state_and_pred_params)

        state: CurrentState = state_and_pred.current_state
        sv_prediction: SVPrediction = state_and_pred.sv_prediction

        rec: rr.RecordingStream = get_recording(state.recording_id)
        stream: rr.BinaryStream = rec.binary_stream()

        current_uvc: Float[ndarray, "n_kpts 3"]
        for idx, current_uvc in enumerate(sv_prediction.uvc_list):
            current_uv: Float[ndarray, "n_kpts 2"] = current_uvc[:, 0:2]  # Extract x, y coordinates
            scores: Float[ndarray, "n_kpts"] = current_uvc[:, -1].astype(np.float32)  # Placeholder for scores
            colors: UInt8[np.ndarray, "n_kpts 3"] = confidence_scores_to_rgb(
                confidence_scores=scores[np.newaxis, :, np.newaxis]
            ).squeeze()
            # set the color of the current keypoint to red
            if state.current_xyxy_idx == idx:
                colors[state.current_uvc_idx, :] = SELECTED_COLOR
            rec.log(
                f"{state.rerun_log_paths['annotated_image_path']}/keypoints_{idx}",
                Points2DWithConfidence(
                    positions=current_uv,
                    confidences=scores.squeeze(),
                    class_ids=2,
                    keypoint_ids=COCO_17_IDS,
                    colors=colors,
                    radii=state.radii,
                ),
            )
        yield stream.read()

    state_and_pred_comps = StateAndPredComponents(
        current_state=current_state,
        sv_prediction=sv_prediction,
    )

    remove_bbox_btn.click(
        fn=remove_bbox_fn,
        inputs=[current_state, sv_prediction],
        outputs=[
            viewer,
            bbox_type,
            current_bbox_text,
            current_state,
            sv_prediction,
        ],
    ).then(
        fn=update_bbox_color,
        inputs=state_and_pred_comps.to_list(),
        outputs=[viewer],
    ).then(
        fn=update_selected_keypoint,
        inputs=state_and_pred_comps.to_list(),
        outputs=[viewer],
    )

    box_btn_left.click(
        fn=lambda state, pred: navigate_bbox(-1, state, pred),
        inputs=[current_state, sv_prediction],
        outputs=[current_bbox_text, current_kpts_name, current_state, sv_prediction],
        queue=False,
    ).then(
        fn=update_bbox_color,
        inputs=state_and_pred_comps.to_list(),
        outputs=[viewer],
    ).then(
        fn=update_selected_keypoint,
        inputs=state_and_pred_comps.to_list(),
        outputs=[viewer],
    )

    box_btn_right.click(
        fn=lambda current_idx, current_list: navigate_bbox(+1, current_idx, current_list),
        inputs=[current_state, sv_prediction],
        outputs=[current_bbox_text, current_kpts_name, current_state, sv_prediction],
        queue=False,
    ).then(
        fn=update_bbox_color,
        inputs=state_and_pred_comps.to_list(),
        outputs=[viewer],
    ).then(
        fn=update_selected_keypoint,
        inputs=state_and_pred_comps.to_list(),
        outputs=[viewer],
    )

    def kpt_step(
        direction: int,
        state: CurrentState,
    ) -> tuple[str, str, CurrentState]:
        """
        Move `idx` left (-1) or right (+1) through OPTIONS with wrap-around.
        Returns the new text and the updated index.
        """
        new_uvc_idx: int = (state.current_uvc_idx + direction) % len(list(COCO_17_NAME2ID.keys()))
        current_kpt_str: str = list(COCO_17_NAME2ID.keys())[new_uvc_idx]

        # Update the current state with the new keypoint index
        state.current_uvc_idx = new_uvc_idx

        return f"Current Keypoint: **{current_kpt_str}**", "Visible", state

    kpt_btn_left.click(
        fn=lambda state: kpt_step(-1, state),
        inputs=[current_state],
        outputs=[current_kpts_name, kpt_visibility, current_state],
    ).then(
        fn=update_selected_keypoint,
        inputs=state_and_pred_comps.to_list(),
        outputs=[viewer],
    )

    kpt_btn_right.click(
        fn=lambda state: kpt_step(+1, state),
        inputs=[current_state],
        outputs=[current_kpts_name, kpt_visibility, current_state],
    ).then(
        fn=update_selected_keypoint,
        inputs=state_and_pred_comps.to_list(),
        outputs=[viewer],
    )

    input_img.change(
        fn=annotate_image,
        inputs=input_components.to_list(),
        outputs=[viewer, sv_prediction, current_state],
    )
