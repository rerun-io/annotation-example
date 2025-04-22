try:
    import spaces  # type: ignore

    IN_SPACES = True
except ImportError:
    print("Not running on Zero")
    IN_SPACES = False


import tempfile
import uuid
from dataclasses import dataclass, fields
from pathlib import Path
from typing import no_type_check

import cv2
import gradio as gr
import numpy as np
import rerun as rr
import rerun.blueprint as rrb
import torch
from einops import rearrange
from gradio_rerun import Rerun
from jaxtyping import Bool, Float, Float32, UInt8
from monopriors.depth_utils import clip_disparity, depth_edges_mask, depth_to_points
from monopriors.relative_depth_models.depth_anything_v2 import (
    DepthAnythingV2Predictor,
    RelativeDepthPrediction,
)
from sam2.sam2_video_predictor import SAM2VideoPredictor
from simplecv.video_io import VideoReader

from annotation_example.gradio_ui.sv_sam_callbacks import RerunLogPaths, single_view_update_keypoints
from annotation_example.gradio_ui.utils import KeypointsContainer, get_recording
from annotation_example.op import create_blueprint

if gr.NO_RELOAD:
    VIDEO_SAM_PREDICTOR: SAM2VideoPredictor = SAM2VideoPredictor.from_pretrained("facebook/sam2-hiera-tiny")
    DEPTH_PREDICTOR = DepthAnythingV2Predictor(device="cpu", encoder="vits")
    DEPTH_PREDICTOR.set_model_device("cuda")


def log_relative_pred_rec(
    rec: rr.RecordingStream,
    parent_log_path: Path,
    relative_pred: RelativeDepthPrediction,
    rgb_hw3: UInt8[np.ndarray, "h w 3"],
    seg_mask_hw: UInt8[np.ndarray, "h w"] | None = None,
    remove_flying_pixels: bool = True,
    jpeg_quality: int = 90,
    depth_edge_threshold: float = 1.1,
) -> None:
    cam_log_path: Path = parent_log_path / "camera"
    pinhole_path: Path = cam_log_path / "pinhole"

    # assume camera is at the origin
    cam_T_world_44: Float[np.ndarray, "4 4"] = np.eye(4)

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
            image_plane_distance=1.5,
            camera_xyz=rr.ViewCoordinates.RDF,
        ),
    )
    rec.log(f"{pinhole_path}/image", rr.Image(rgb_hw3).compress(jpeg_quality=jpeg_quality))

    depth_hw: Float32[np.ndarray, "h w"] = relative_pred.depth
    disparity = relative_pred.disparity
    # removes outliers from disparity (sometimes we can get weirdly large values)
    clipped_disparity: UInt8[np.ndarray, "h w"] = clip_disparity(disparity)
    if remove_flying_pixels:
        edges_mask: Bool[np.ndarray, "h w"] = depth_edges_mask(depth_hw, threshold=depth_edge_threshold)
        rec.log(
            f"{pinhole_path}/edge_mask",
            rr.SegmentationImage(edges_mask.astype(np.uint8)),
        )
        depth_hw: Float32[np.ndarray, "h w"] = depth_hw * ~edges_mask
        clipped_disparity: UInt8[np.ndarray, "h w"] = clipped_disparity * ~edges_mask

    if seg_mask_hw is not None:
        rec.log(
            f"{pinhole_path}/segmentation",
            rr.SegmentationImage(seg_mask_hw),
        )
        depth_hw: Float32[np.ndarray, "h w"] = depth_hw  # * seg_mask_hw
        clipped_disparity: UInt8[np.ndarray, "h w"] = clipped_disparity  # * seg_mask_hw

    rec.log(f"{pinhole_path}/depth", rr.DepthImage(depth_hw))

    # log to cam_log_path to avoid backprojecting disparity
    rec.log(f"{cam_log_path}/disparity", rr.DepthImage(clipped_disparity))

    depth_1hw: Float32[np.ndarray, "1 h w"] = rearrange(depth_hw, "h w -> 1 h w")
    pts_3d: Float32[np.ndarray, "h w 3"] = depth_to_points(depth_1hw, relative_pred.K_33)

    colors = rgb_hw3.reshape(-1, 3)

    # If we have a segmentation mask, make those pixels blue
    if seg_mask_hw is not None:
        # Reshape the mask to match colors shape
        flat_mask = seg_mask_hw.reshape(-1)

        # Set pixels where mask == 1 to blue (BGR format)
        # Blue: [255, 0, 0] in BGR or [0, 0, 255] in RGB
        colors[flat_mask == 1, :] = [0, 0, 255]  # RGB format: Blue

    rec.log(
        f"{parent_log_path}/point_cloud",
        rr.Points3D(
            positions=pts_3d.reshape(-1, 3),
            colors=colors,
        ),
    )


def rescale_img(img_hw3: UInt8[np.ndarray, "h w 3"], max_dim: int) -> UInt8[np.ndarray, "... 3"]:
    # resize the image to have a max dim of max_dim
    height, width, _ = img_hw3.shape
    current_dim = max(height, width)

    # If current dimension is larger than max_dim, calculate scale factor
    if current_dim > max_dim:
        scale_factor = max_dim / current_dim
        new_height = int(height * scale_factor)
        new_width = int(width * scale_factor)

        # Resize image maintaining aspect ratio
        resized_img: UInt8[np.ndarray, "... 3"] = cv2.resize(
            img_hw3, (new_width, new_height), interpolation=cv2.INTER_AREA
        )
        return resized_img

    # Return original image if no resize needed
    return img_hw3


# Allow using keyword args in gradio to avoid mixing up the order of inputs
# a bit of an antipattern that is requied to make things work with beartype + keyword args
@dataclass
class PreprocessVideoComponents:
    video_file: gr.Video

    def to_list(self) -> list:
        return [getattr(self, f.name) for f in fields(self)]


@dataclass
class PreprocessVideoValues:
    video_file: str


@no_type_check
def preprocess_video(
    *preprocess_video_params,
):
    yield from _preprocess_video(*preprocess_video_params)


def _preprocess_video(
    *input_params,
    progress=gr.Progress(track_tqdm=True),  # noqa B008
):
    input_values: PreprocessVideoValues = PreprocessVideoValues(*input_params)
    # create a new recording id, and store it in a Gradio's session state.
    recording_id: uuid.UUID = uuid.uuid4()
    rec: rr.RecordingStream = get_recording(recording_id)
    stream: rr.BinaryStream = rec.binary_stream()

    log_paths = RerunLogPaths(
        timeline_name="frame_idx",
        parent_log_path=Path("world"),
        camera_log_path=Path("world") / "camera",
        pinhole_path=Path("world") / "camera" / "pinhole",
    )

    video_path: Path = Path(input_values.video_file)

    initial_blueprint = rrb.Blueprint(
        rrb.Horizontal(
            rrb.Spatial2DView(origin=f"{log_paths['pinhole_path']}"),
        ),
        collapse_panels=True,
    )

    rec.send_blueprint(initial_blueprint)

    video_reader: VideoReader = VideoReader(video_path)
    tmp_frames_dir: str = tempfile.mkdtemp()

    target_fps: int = 10
    frame_interval: int = int(video_reader.fps // target_fps)
    max_frames: int = 100
    total_saved_frames: int = 0
    max_size: int = 640

    progress(0, desc="Reading video frames")
    for idx, bgr in enumerate(video_reader):
        if idx % frame_interval == 0:
            if total_saved_frames >= max_frames:
                break
            bgr: np.ndarray = rescale_img(bgr, max_size)
            # 3. Save frames to temporary directory
            cv2.imwrite(f"{tmp_frames_dir}/{idx:05d}.jpg", bgr)
            total_saved_frames += 1

    first_frame_path: Path = Path(tmp_frames_dir) / "00000.jpg"
    first_bgr: np.ndarray = cv2.imread(str(first_frame_path))

    progress(0.5, desc="Initializing SAM")
    with torch.inference_mode():
        inference_state = VIDEO_SAM_PREDICTOR.init_state(video_path=tmp_frames_dir)
        VIDEO_SAM_PREDICTOR.reset_state(inference_state)
    print(type(inference_state))

    rec.set_time_sequence(log_paths["timeline_name"], sequence=0)
    rec.log(
        f"{log_paths['pinhole_path']}/image",
        rr.Image(first_bgr, color_model=rr.ColorModel.BGR).compress(jpeg_quality=90),
    )

    # Ensure we consume everything from the recording.
    stream.flush()

    yield gr.Accordion(open=False), stream.read(), inference_state, Path(tmp_frames_dir), recording_id, log_paths


@no_type_check
def reset_keypoints(
    active_recording_id: uuid.UUID,
    keypoints_container: KeypointsContainer,
    log_paths: RerunLogPaths,
):
    yield from _reset_keypoints(active_recording_id, keypoints_container, log_paths)


def _reset_keypoints(active_recording_id: uuid.UUID, keypoints_container: KeypointsContainer, log_paths: RerunLogPaths):
    # Now we can produce a valid keypoint.
    rec: rr.RecordingStream = get_recording(active_recording_id)
    stream: rr.BinaryStream = rec.binary_stream()

    keypoints_container.clear()

    rec.set_time_sequence(log_paths["timeline_name"], sequence=0)
    rec.log(
        f"{log_paths['pinhole_path']}/image/include",
        rr.Clear(recursive=True),
    )
    rec.log(
        f"{log_paths['pinhole_path']}/image/exclude",
        rr.Clear(recursive=True),
    )
    rec.log(
        f"{log_paths['pinhole_path']}/segmentation",
        rr.Clear(recursive=True),
    )
    rec.log(
        f"{log_paths['pinhole_path']}/depth",
        rr.Clear(recursive=True),
    )

    # Ensure we consume everything from the recording.
    stream.flush()
    yield stream.read(), keypoints_container


@no_type_check
def get_initial_mask(
    recording_id: uuid.UUID,
    inference_state: dict,
    keypoint_container: KeypointsContainer,
    log_paths: RerunLogPaths,
):
    yield from _get_initial_mask(recording_id, inference_state, keypoint_container, log_paths)


def _get_initial_mask(
    recording_id: uuid.UUID,
    inference_state: dict,
    keypoint_container: KeypointsContainer,
    log_paths: RerunLogPaths,
):
    rec = get_recording(recording_id)
    stream = rec.binary_stream()

    rec.set_time_sequence(log_paths["timeline_name"], 0)

    points = np.vstack([keypoint_container.include_points, keypoint_container.exclude_points]).astype(np.float32)
    if len(points) == 0:
        raise gr.Error("No points selected. Please add include or exclude points.")

    # Create labels array: 1 for include points, 0 for exclude points
    labels = np.ones(len(keypoint_container.include_points), dtype=np.int32)
    if len(keypoint_container.exclude_points) > 0:
        labels = np.concatenate([labels, np.zeros(len(keypoint_container.exclude_points), dtype=np.int32)])

    print(f"Points shape: {points.shape}")
    print(f"Labels shape: {labels.shape}")
    print(labels)
    print(
        f"Include points: {keypoint_container.include_points.shape}, Exclude points: {keypoint_container.exclude_points.shape}"
    )

    with torch.inference_mode():
        frame_idx: int
        object_ids: list
        masks: Float32[torch.Tensor, "b 3 h w"]

        frame_idx, object_ids, masks = VIDEO_SAM_PREDICTOR.add_new_points_or_box(
            inference_state=inference_state,
            frame_idx=0,
            obj_id=0,
            points=points,
            labels=labels,
        )

        masks: Bool[np.ndarray, "1 h w"] = (masks[0] > 0.0).numpy(force=True)

    rec.log(
        f"{log_paths['pinhole_path']}/segmentation",
        rr.SegmentationImage(masks[0].astype(np.uint8)),
    )
    yield stream.read()


@no_type_check
def propagate_mask(
    recording_id: uuid.UUID,
    inference_state: dict,
    keypoint_container: KeypointsContainer,
    frames_dir: Path,
    log_paths: RerunLogPaths,
):
    yield from _propagate_mask(recording_id, inference_state, keypoint_container, frames_dir, log_paths)


def _propagate_mask(
    recording_id: uuid.UUID,
    inference_state: dict,
    keypoint_container: KeypointsContainer,
    frames_dir: Path,
    log_paths: RerunLogPaths,
):
    rec = get_recording(recording_id)
    stream = rec.binary_stream()

    blueprint = create_blueprint(parent_log_path=log_paths["parent_log_path"])
    rec.send_blueprint(blueprint)

    rec.log(f"{log_paths['parent_log_path']}", rr.ViewCoordinates.RDF)

    points = np.vstack([keypoint_container.include_points, keypoint_container.exclude_points]).astype(np.float32)
    if len(points) == 0:
        raise gr.Error("No points selected. Please add include or exclude points.")

    # Create labels array: 1 for include points, 0 for exclude points
    labels = np.ones(len(keypoint_container.include_points), dtype=np.int32)
    if len(keypoint_container.exclude_points) > 0:
        labels = np.concatenate([labels, np.zeros(len(keypoint_container.exclude_points), dtype=np.int32)])

    frames_paths: list[Path] = sorted(frames_dir.glob("*.jpg"))

    # remove the keypoints as they're in the way during propagation
    rec.log(
        f"{log_paths['pinhole_path']}/include",
        rr.Clear(recursive=True),
    )
    rec.log(
        f"{log_paths['pinhole_path']}/exclude",
        rr.Clear(recursive=True),
    )

    with torch.inference_mode():
        frame_idx: int
        object_ids: list
        masks: Float32[torch.Tensor, "b 3 h w"]

        frame_idx, object_ids, masks = VIDEO_SAM_PREDICTOR.add_new_points_or_box(
            inference_state, frame_idx=0, obj_id=0, points=points, labels=labels
        )

        # propagate the prompts to get masklets throughout the video
        for frames_path, (frame_idx, object_ids, masks) in zip(
            frames_paths, VIDEO_SAM_PREDICTOR.propagate_in_video(inference_state), strict=True
        ):
            rec.set_time_sequence(log_paths["timeline_name"], frame_idx)
            masks: Bool[np.ndarray, "1 h w"] = (masks[0] > 0.0).numpy(force=True)
            bgr = cv2.imread(str(frames_path))
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            depth_pred: RelativeDepthPrediction = DEPTH_PREDICTOR.__call__(rgb=rgb, K_33=None)

            log_relative_pred_rec(
                rec=rec,
                parent_log_path=log_paths["parent_log_path"],
                relative_pred=depth_pred,
                rgb_hw3=rgb,
                seg_mask_hw=masks[0].astype(np.uint8),
                remove_flying_pixels=True,
                jpeg_quality=90,
                depth_edge_threshold=0.1,
            )

            yield stream.read()


with gr.Blocks() as single_view_block:
    keypoints = gr.State(KeypointsContainer.empty())
    inference_state = gr.State({})
    frames_dir = gr.State(Path())
    with gr.Row():
        with gr.Column(scale=1):
            with gr.Accordion("Your video IN", open=True) as video_in_drawer:
                video_in = gr.Video(label="Video IN", format=None)

            point_type = gr.Radio(
                label="point type",
                choices=["include", "exclude"],
                value="include",
                scale=1,
            )
            reset_points_btn = gr.Button("Clear Points", scale=1)
            get_initial_mask_btn = gr.Button("Get Initial Mask", scale=1)
            propagate_mask_btn = gr.Button("Propagate Mask", scale=1)
            stop_propagation_btn = gr.Button("Stop Propagation", scale=1)

        with gr.Column(scale=4):
            viewer = Rerun(
                streaming=True,
                panel_states={
                    "time": "collapsed",
                    "blueprint": "hidden",
                    "selection": "hidden",
                },
                height=700,
            )

    # We make a new recording id, and store it in a Gradio's session state.
    recording_id = gr.State()
    log_paths = gr.State({})

    preprocess_video_comps = PreprocessVideoComponents(
        video_file=video_in,
    )

    # triggered on video upload
    video_in.upload(
        fn=preprocess_video,
        inputs=preprocess_video_comps.to_list(),
        outputs=[video_in_drawer, viewer, inference_state, frames_dir, recording_id, log_paths],
    )

    viewer.selection_change(
        single_view_update_keypoints,
        inputs=[
            recording_id,
            point_type,
            keypoints,
            log_paths,
        ],
        outputs=[viewer, keypoints],
    )

    reset_points_btn.click(
        fn=reset_keypoints,
        inputs=[recording_id, keypoints, log_paths],
        outputs=[viewer, keypoints],
    )

    get_initial_mask_btn.click(
        fn=get_initial_mask,
        inputs=[recording_id, inference_state, keypoints, log_paths],
        outputs=[viewer],
    )

    propagate_event = propagate_mask_btn.click(
        fn=propagate_mask,
        inputs=[recording_id, inference_state, keypoints, frames_dir, log_paths],
        outputs=[viewer],
    )

    stop_propagation_btn.click(
        fn=lambda: None,
        inputs=[],
        outputs=[],
        cancels=[propagate_event],
    )
