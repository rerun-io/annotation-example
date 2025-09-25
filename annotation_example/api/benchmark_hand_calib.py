from dataclasses import dataclass, field
from pathlib import Path
from timeit import default_timer as timer
from typing import Literal, NamedTuple

import cv2
import numpy as np
import rerun as rr
import rerun.blueprint as rrb
from jaxopt._src.levenberg_marquardt import LevenbergMarquardtState
from jaxtyping import Float, Float32, Int, UInt8
from numpy import ndarray
from simplecv.apis.view_exoego import (
    LogPaths,
    SceneSetupResult,
    compute_vertex_normals_batch,
    create_blueprint,
    log_exoego_batch,
    setup_scene,
)
from simplecv.camera_parameters import PinholeParameters
from simplecv.configs.exoego_dataset_configs import AnnotatedExoEgoDatasetUnion
from simplecv.data.exo.base_exo import BaseExoSequence
from simplecv.data.exoego.base_exoego import BaseExoEgoSequence
from simplecv.data.skeleton.coco_133 import (
    COCO_133_ID2NAME,
    COCO_133_IDS,
    COCO_133_LINKS,
    LEFT_HAND_IDX,
    RIGHT_HAND_IDX,
)
from simplecv.data.skeleton.mediapipe import MEDIAPIPE_ID2NAME, MEDIAPIPE_IDS, MEDIAPIPE_LINKS
from simplecv.ops.mano.mano_np import MANOLayerNP
from simplecv.ops.mano.optim_jax_single_shape import (
    OptimShapeInput,
    OptimShapeResult,
    PoseShapeOptimConfig,
    SingleHandShapeOptim,
)
from simplecv.ops.triangulate import batch_triangulate
from simplecv.rerun_log_utils import (
    Points2DWithConfidence,
    Points3DWithConfidence,
    RerunTyroConfig,
    confidence_scores_to_rgb,
)
from simplecv.video_io import MultiVideoReader
from wilor_nano.hand_detection import DetectionResult, HandDetector, HandDetectorConfig
from wilor_nano.hand_keypoints import (
    HandKeypointDetectorConfig,
    KeypointResults,
    WilorHandKeypointDetector,
)


def timestamp_to_frame_index(time_ns: int, frame_timestamps_ns: Int[ndarray, "num_frames"]) -> int:
    """Return the frame index at or before ``time_ns`` for monotonic timestamps."""

    idx: int = int(np.searchsorted(frame_timestamps_ns, time_ns, side="right") - 1)
    return max(0, min(idx, len(frame_timestamps_ns) - 1))


def frame_index_to_timestamp(frame_timestamps_ns: Int[ndarray, "num_frames"], frame_index: int) -> int:
    """Return the nanosecond timestamp associated with ``frame_index``."""

    if frame_index < 0 or frame_index >= int(frame_timestamps_ns.shape[0]):
        msg = f"frame_index {frame_index} is outside the valid range [0, {frame_timestamps_ns.shape[0] - 1}]"
        raise IndexError(msg)
    timestamp_ns: int = int(frame_timestamps_ns[frame_index])
    return timestamp_ns


def set_annotation_context(recording: rr.RecordingStream | None = None) -> None:
    """Register Mediapipe (per-hand) and COCO-133 annotation metadata with Rerun."""

    rr.log(
        "/",
        rr.AnnotationContext(
            [
                rr.ClassDescription(
                    info=rr.AnnotationInfo(id=0, label="L", color=(0, 0, 255)),
                    keypoint_annotations=[
                        rr.AnnotationInfo(id=kp_id, label=name) for kp_id, name in MEDIAPIPE_ID2NAME.items()
                    ],
                    keypoint_connections=MEDIAPIPE_LINKS,
                ),
                rr.ClassDescription(
                    info=rr.AnnotationInfo(id=1, label="R", color=(255, 0, 0)),
                    keypoint_annotations=[
                        rr.AnnotationInfo(id=kp_id, label=name) for kp_id, name in MEDIAPIPE_ID2NAME.items()
                    ],
                    keypoint_connections=MEDIAPIPE_LINKS,
                ),
                rr.ClassDescription(
                    info=rr.AnnotationInfo(id=2, label="Coco Wholebody", color=(0, 0, 255)),
                    keypoint_annotations=[
                        rr.AnnotationInfo(id=kp_id, label=name) for kp_id, name in COCO_133_ID2NAME.items()
                    ],
                    keypoint_connections=COCO_133_LINKS,
                ),
            ]
        ),
        static=True,
        recording=recording,
    )


@dataclass
class HandCalibratorConfig:
    """Configuration parameters steering multi-view MANO optimisation (pipeline step 3)."""

    hand_side: Literal["left", "right"] = "right"
    """Which hand to optimize with MANO—either "left" or "right"."""
    detection_confidence: float = 0.3
    """Minimum detector confidence required to keep a hand bounding box."""
    ts_nano: int | None = None
    """Optional absolute timestamp in nanoseconds to sample; defaults to first frame when ``None``."""
    mano_optim_iters: int = 30
    """Number of Levenberg-Marquardt iterations for MANO pose/shape fitting."""
    n_frame_optim: int = 1
    """Number of consecutive frames to jointly optimize."""
    verbose: bool = True
    """Whether to log additional intermediate results for debugging purposes."""


@dataclass
class HandCalibrationResult:
    """Outputs from a single calibration pass for downstream consumers."""

    pinhole_param_list: list[PinholeParameters]
    """Per-camera pinhole parameters aligned with the exo rig ordering."""
    xyz: Float[ndarray, "n_kpts=133 3"]
    """Triangulated 3D COCO-133 keypoints at the calibrated timestamp."""
    confidences: Float[ndarray, "n_kpts=133"]
    """Confidence score per COCO-133 keypoint derived from multi-view triangulation."""
    # mano_vertices: Float32[ndarray, "n_verts=778 3"]
    # """Optimized MANO vertex positions for the requested hand."""
    # mano_joints: Float32[ndarray, "n_joints=21 3"]
    # """Optimized MANO joint positions matching Mediapipe ordering."""
    # mano_faces: Int[ndarray, "n_faces=1538 3"]
    # """Triangle indices describing the MANO mesh topology."""


class ParsedDetections(NamedTuple):
    """Convenience container bundling multi-view detections for triangulation."""

    uvc_coco_batch: Float[ndarray, "n_views n_kpts=133 3"]
    right_hand_kpts: KeypointResults | None
    left_hand_kpts: KeypointResults | None


class HandCalibrator:
    def __init__(
        self,
        hand_detector: HandDetector,
        hand_keypoint_detector: WilorHandKeypointDetector,
        config: HandCalibratorConfig,
    ) -> None:
        self.hand_detector: HandDetector = hand_detector
        self.hand_keypoint_detector: WilorHandKeypointDetector = hand_keypoint_detector
        self.config: HandCalibratorConfig = config

    def __call__(
        self,
        *,
        exo_cam_list: list[PinholeParameters],
        rgb_ts_batch: UInt8[ndarray, "n_frames n_views H W 3"],
        parent_log_path: Path,
    ) -> HandCalibrationResult:
        calibration_root: Path = parent_log_path / "hand_calibration"

        rgb_list: UInt8[ndarray, "n_views H W 3"] = rgb_ts_batch[0]

        parsed_detections: ParsedDetections = self._detect_keypoints(
            rgb_list=rgb_list,
            pinhole_param_list=exo_cam_list,
            parent_log_path=parent_log_path,
        )

        triangulation_result: tuple[
            Float[ndarray, "n_kpts=133 3"],
            Float[ndarray, "n_kpts=133"],
        ] = self._triangulate_keypoints(
            parsed_detections.uvc_coco_batch,
            exo_cam_list,
            calibration_root=calibration_root,
        )
        xyz: Float[ndarray, "n_kpts=133 3"]
        conf_values: Float[ndarray, "n_kpts=133"]
        xyz, conf_values = triangulation_result

        mano_vertices: Float32[ndarray, "n_verts=778 3"]
        mano_joints: Float32[ndarray, "n_joints=21 3"]
        mano_faces: Int[ndarray, "n_faces=1538 3"]
        mano_vertices, mano_joints, mano_faces = self._optimize_mano(
            xyz=xyz,
            confidences=conf_values,
            parsed_detections=parsed_detections,
            pinhole_param_list=exo_cam_list,
            calibration_root=calibration_root,
        )

        return HandCalibrationResult(
            pinhole_param_list=exo_cam_list,
            xyz=xyz,
            confidences=conf_values,
            # mano_vertices=mano_vertices,
            # mano_joints=mano_joints,
            # mano_faces=mano_faces,
        )

    def _detect_keypoints(
        self,
        *,
        rgb_list: UInt8[ndarray, "n_views H W 3"],
        pinhole_param_list: list[PinholeParameters],
        parent_log_path: Path,
    ) -> ParsedDetections:
        uvc_coco_list: list[Float[ndarray, "n_kpts=133 3"]] = []
        right_hand_kpts: KeypointResults | None = None
        left_hand_kpts: KeypointResults | None = None

        for camera_idx, (rgb_hw3, pinhole) in enumerate(zip(rgb_list, pinhole_param_list, strict=True)):
            det_result: DetectionResult = self.hand_detector(
                rgb_hw3=rgb_hw3,
                hand_conf=self.config.detection_confidence,
            )
            camera_name: str = getattr(pinhole, "name", f"camera_{camera_idx}")
            hand_path_root: Path = parent_log_path / "exo" / camera_name / "pinhole"
            uvc_coco: Float[ndarray, "n_kpts=133 3"] = np.full((133, 3), np.nan, dtype=np.float32)

            for hand_label, xyxy in ("left", det_result.left_xyxy), ("right", det_result.right_xyxy):
                if xyxy is None:
                    continue

                hand_log_path: Path = hand_path_root / "video" / hand_label
                kpts_results: KeypointResults = self.hand_keypoint_detector(
                    rgb_hw3=rgb_hw3,
                    xyxy=xyxy,
                    handedness=hand_label,
                )
                uv: Float[ndarray, "n_frames=1 n_kpts=21 2"] = kpts_results.keypoints_2d
                conf: Float[ndarray, "n_frames=1 n_kpts=21"] = kpts_results.scores
                conf_values: Float[ndarray, "n_kpts=21"] = conf[0].astype(np.float32)
                mean_conf: float = float(np.nanmean(conf_values))
                uv_filtered: Float[ndarray, "n_kpts=21 2"] = uv[0].copy()
                if mean_conf < 0.4:
                    uv_filtered[:] = np.nan
                    conf_values[:] = 0.0

                conf_colors: UInt8[ndarray, "n_frames=1 n_kpts=21 3"] = confidence_scores_to_rgb(
                    confidence_scores=conf_values[np.newaxis, :, np.newaxis]
                )

                fill_idx = RIGHT_HAND_IDX if hand_label == "right" else LEFT_HAND_IDX
                uvc_coco[fill_idx, :2] = uv_filtered
                uvc_coco[fill_idx, 2] = conf_values
                if hand_label == "right" and right_hand_kpts is None:
                    right_hand_kpts = kpts_results
                if hand_label == "left" and left_hand_kpts is None:
                    left_hand_kpts = kpts_results

                rr.log(
                    f"{hand_log_path}/bbox",
                    rr.Boxes2D(
                        array=xyxy,
                        array_format=rr.Box2DFormat.XYXY,
                        class_ids=0 if hand_label == "left" else 1,
                        show_labels=True,
                    ),
                )
                rr.log(
                    f"{hand_log_path}/keypoints",
                    Points2DWithConfidence(
                        positions=uv_filtered,
                        confidences=conf_values,
                        class_ids=0 if hand_label == "left" else 1,
                        keypoint_ids=MEDIAPIPE_IDS,
                        show_labels=False,
                        colors=conf_colors[0],
                    ),
                )

            uvc_coco_list.append(uvc_coco)

        uvc_coco_batch: Float[ndarray, "n_views n_kpts=133 3"] = np.stack(uvc_coco_list)
        return ParsedDetections(
            uvc_coco_batch=uvc_coco_batch,
            right_hand_kpts=right_hand_kpts,
            left_hand_kpts=left_hand_kpts,
        )

    def _triangulate_keypoints(
        self,
        uvc_coco_batch: Float[ndarray, "n_views n_kpts=133 3"],
        pinhole_param_list: list[PinholeParameters],
        *,
        calibration_root: Path,
    ) -> tuple[Float[ndarray, "n_kpts=133 3"], Float[ndarray, "n_kpts=133"]]:
        uvc_triangulate_batch: Float[ndarray, "n_views n_kpts=133 3"] = np.nan_to_num(
            uvc_coco_batch.copy(),
            nan=0.0,
        )
        Pall_exo: Float[ndarray, "n_views 3 4"] = np.stack(
            [pinhole.projection_matrix for pinhole in pinhole_param_list]
        ).astype(np.float32)
        xyzc: Float[ndarray, "n_kpts=133 4"] = batch_triangulate(
            uvc_triangulate_batch,
            Pall_exo,
            min_views=2,
        )
        xyz: Float[ndarray, "n_kpts=133 3"] = xyzc[:, :3]
        conf_values: Float[ndarray, "n_kpts=133"] = xyzc[:, 3]
        xyz_for_logging: Float[ndarray, "n_kpts=133 3"] = np.where(
            conf_values[:, np.newaxis] > 0,
            xyz,
            np.nan,
        )
        conf_colors: UInt8[ndarray, "1 n_kpts=133 3"] = confidence_scores_to_rgb(
            conf_values[:, np.newaxis][np.newaxis, ...]
        )

        rr.log(
            f"{calibration_root}/wb_keypoints",
            Points3DWithConfidence(
                positions=xyz_for_logging,
                confidences=conf_values,
                class_ids=2,
                keypoint_ids=COCO_133_IDS,
                show_labels=False,
                colors=conf_colors[0],
            ),
        )
        return xyz, conf_values

    def _optimize_mano(
        self,
        *,
        xyz: Float[ndarray, "n_kpts=133 3"],
        confidences: Float[ndarray, "n_kpts=133"],
        parsed_detections: ParsedDetections,
        pinhole_param_list: list[PinholeParameters],
        calibration_root: Path,
    ) -> tuple[
        Float32[ndarray, "n_verts=778 3"],
        Float32[ndarray, "n_joints=21 3"],
        Int[ndarray, "n_faces=1538 3"],
    ]:
        hand_side: Literal["left", "right"] = self.config.hand_side
        n_frames_optim: int = 1
        uv_exo_stack: Float[ndarray, "n_frames=1 n_views n_kpts=133 2"] = parsed_detections.uvc_coco_batch[
            np.newaxis, :, :, 0:2
        ]

        Pall_exo: Float[ndarray, "n_views 3 4"] = np.stack(
            [pinhole.projection_matrix for pinhole in pinhole_param_list]
        ).astype(np.float32)

        kpts_results_selected: KeypointResults | None = (
            parsed_detections.right_hand_kpts if hand_side == "right" else parsed_detections.left_hand_kpts
        )
        if kpts_results_selected is None:
            raise ValueError(f"No keypoint detections available for hand '{hand_side}'")

        if kpts_results_selected.global_orient is None or kpts_results_selected.hand_pose is None:
            so3_init: Float[ndarray, "b 48"] = np.zeros((n_frames_optim, 48), dtype=np.float32)
        else:
            global_orient: Float[ndarray, "b 1 3"] = kpts_results_selected.global_orient.astype(np.float32)
            hand_pose: Float[ndarray, "b 15 3"] = kpts_results_selected.hand_pose.astype(np.float32)
            so3_concat: Float[ndarray, "b 16 3"] = np.concatenate([global_orient, hand_pose], axis=1)
            so3_init = so3_concat.reshape(so3_concat.shape[0], -1)
            if so3_init.shape[0] != n_frames_optim:
                if so3_init.shape[0] == 0:
                    so3_init = np.zeros((n_frames_optim, 48), dtype=np.float32)
                else:
                    so3_init = np.broadcast_to(so3_init[0:1], (n_frames_optim, so3_init.shape[1])).astype(
                        np.float32,
                        copy=True,
                    )
            else:
                so3_init = so3_init.astype(np.float32, copy=False)

        hand_indices = RIGHT_HAND_IDX if hand_side == "right" else LEFT_HAND_IDX
        hand_xyz: Float[ndarray, "n_hand 3"] = xyz[hand_indices]
        hand_conf: Float[ndarray, "n_hand"] = confidences[hand_indices]
        valid_hand_mask: np.ndarray = hand_conf > 0.0
        valid_indices: Int[ndarray, "n_valid"] = np.where(valid_hand_mask)[0]
        if np.any(valid_hand_mask):
            trans_guess: Float[ndarray, "3"] = hand_xyz[valid_hand_mask].mean(axis=0).astype(np.float32)
        else:
            trans_guess = np.zeros((3,), dtype=np.float32)
        valid_indices: Int[ndarray, "n_valid"] = np.where(valid_hand_mask)[0]
        trans_init: Float[ndarray, "b 3"] = np.broadcast_to(trans_guess, (n_frames_optim, 3)).astype(
            np.float32, copy=True
        )

        optim_shape_cfg = PoseShapeOptimConfig(
            Pall=Pall_exo,
            hand_side=hand_side,
            n_frames_optim=n_frames_optim,
            n_optim_iters=self.config.mano_optim_iters,
        )
        optimizer_shape = SingleHandShapeOptim(config=optim_shape_cfg)

        if kpts_results_selected.betas is not None and kpts_results_selected.betas.size > 0:
            betas_array = np.asarray(kpts_results_selected.betas, dtype=np.float32)
            if betas_array.ndim == 1:
                beta_init: Float[ndarray, "10"] = betas_array
            else:
                beta_init = betas_array[0]
        else:
            beta_init = np.zeros((10,), dtype=np.float32)

        optim_shape_input: OptimShapeInput = OptimShapeInput(
            uv_pred=uv_exo_stack[:n_frames_optim],
            beta_init=beta_init,
            so3_init=so3_init,
            trans_init=trans_init,
        )
        optim_shape_tuple: tuple[OptimShapeResult, LevenbergMarquardtState] = optimizer_shape(optim_shape_input)
        optim_shape_result: OptimShapeResult = optim_shape_tuple[0]

        beta_optim: Float[ndarray, "10"] = optim_shape_result.beta_optim.astype(np.float32, copy=False)

        mano_init_layer = MANOLayerNP(side=hand_side, betas=beta_init.astype(np.float32, copy=False))
        mano_optim_layer = MANOLayerNP(side=hand_side, betas=beta_optim)

        verts_init, joints_init = mano_init_layer(
            so3_init.astype(np.float32, copy=False),
            trans_init.astype(np.float32, copy=False),
        )
        verts_optim, joints_optim = mano_optim_layer(
            optim_shape_result.so3_optim.astype(np.float32, copy=False),
            optim_shape_result.trans_optim.astype(np.float32, copy=False),
        )

        faces_np: Int[ndarray, "n_faces=1538 3"] = mano_optim_layer.f.astype(np.int32)
        normals_init: Float32[ndarray, "n_frames n_verts=778 3"] = compute_vertex_normals_batch(
            verts_init[0:1], faces_np
        )
        normals_optim: Float32[ndarray, "n_frames n_verts=778 3"] = compute_vertex_normals_batch(
            verts_optim[0:1], faces_np
        )

        mano_mesh_path: Path = calibration_root / f"{hand_side}_mano_mesh"
        mano_joint_path: Path = calibration_root / f"{hand_side}_mano_xyz"

        triangulated_hand: Float[ndarray, "n_joints=21 3"] = xyz[hand_indices]
        verts_aligned_init: Float32[ndarray, "n_verts=778 3"] = verts_init[0]
        joints_aligned_init: Float32[ndarray, "n_joints=21 3"] = joints_init[0]
        verts_aligned_optim: Float32[ndarray, "n_verts=778 3"] = verts_optim[0]
        joints_aligned_optim: Float32[ndarray, "n_joints=21 3"] = joints_optim[0]

        if self.config.verbose:
            rr.log(
                f"{mano_mesh_path}_init",
                rr.Mesh3D(
                    vertex_positions=verts_aligned_init,
                    triangle_indices=faces_np,
                    vertex_normals=normals_init[0],
                    albedo_factor=(255, 0, 0, 255),
                ),
            )
        rr.log(
            f"{mano_mesh_path}_optim",
            rr.Mesh3D(
                vertex_positions=verts_aligned_optim,
                triangle_indices=faces_np,
                vertex_normals=normals_optim[0],
                albedo_factor=(0, 0, 255, 255),
            ),
        )
        class_id: int = 1 if hand_side == "right" else 0
        class_ids: Int[ndarray, "n_joints=21"] = np.full((joints_aligned_optim.shape[0],), class_id, dtype=np.int32)
        keypoint_ids: Int[ndarray, "n_joints=21"] = np.asarray(MEDIAPIPE_IDS, dtype=np.int32)
        rr.log(
            f"{mano_joint_path}",
            rr.Points3D(
                joints_aligned_optim,
                class_ids=class_ids,
                keypoint_ids=keypoint_ids,
                show_labels=False,
            ),
        )

        return verts_aligned_optim, joints_aligned_optim, faces_np


@dataclass
class BenchmarkHandCalibConfig:
    """CLI configuration controlling dataset selection and calibration behaviour."""

    rr_config: RerunTyroConfig
    """Viewer launch configuration propagated to the Rerun harness."""
    dataset: AnnotatedExoEgoDatasetUnion
    """Dataset specification defining which ego/exo sequence to process."""
    log_labels: bool = False
    """Whether to stream ground-truth labels alongside calibration outputs."""
    hand_calibrator: HandCalibratorConfig = field(default_factory=HandCalibratorConfig)
    """Parameter bundle forwarded to the ``HandCalibrator`` instance."""


def mv_reader_to_rgb_ts_batch(
    mv_reader: MultiVideoReader,
    num_frames: int,
    ts_nanos: int,
    frame_timestamps_ns: Int[ndarray, "n_frames"],
) -> UInt8[ndarray, "n_frames n_views H W 3"]:
    """Slice a timestamp-aligned RGB batch from a ``MultiVideoReader``.

    Args:
        mv_reader: Multi-camera video reader covering the synchronized views.
        num_frames: Number of consecutive frames to fetch, inclusive of the
            frame containing ``ts_nanos``.
        ts_nanos: Absolute nanosecond timestamp selecting the first frame.
        frame_timestamps_ns: Shared monotonic timestamp vector used to align
            frame indices across views (commonly ``SceneSetupResult.shortest_timestamp``).

    Returns:
        ``UInt8[np.ndarray, "n_frames n_views H W 3"]`` containing the RGB
        frames ordered by increasing timestamp and view index.

    Raises:
        ValueError: If there are no views, no frames, ``num_frames`` is not
            positive, or the requested batch exceeds the available frames from
            ``ts_nanos`` onward.
    """

    if num_frames <= 0:
        raise ValueError("num_frames must be a positive integer")

    total_frames: int = len(mv_reader)
    n_views: int = len(mv_reader.video_readers)
    if total_frames == 0 or n_views == 0:
        raise ValueError("MultiVideoReader contains no frames or no views")

    start_idx: int = timestamp_to_frame_index(ts_nanos, frame_timestamps_ns)
    max_available_frames: int = total_frames - start_idx
    if max_available_frames < num_frames:
        raise ValueError("Requested number of frames exceeds available frames from the provided timestamp")

    rgb_frames: list[UInt8[ndarray, "n_views H W 3"]] = []
    for frame_offset in range(num_frames):
        frame_idx: int = start_idx + frame_offset
        bgr_list: list[UInt8[ndarray, "H W 3"]] = mv_reader[frame_idx]
        rgb_views: list[UInt8[ndarray, "H W 3"]] = [cv2.cvtColor(bgr_hw3, cv2.COLOR_BGR2RGB) for bgr_hw3 in bgr_list]
        rgb_views_stack: UInt8[ndarray, "n_views H W 3"] = np.stack(rgb_views, axis=0)
        rgb_frames.append(rgb_views_stack)

    rgb_ts_batch: UInt8[ndarray, "n_frames n_views H W 3"] = np.stack(rgb_frames, axis=0)
    return rgb_ts_batch


def main(config: BenchmarkHandCalibConfig) -> None:
    start_time: float = timer()
    exoego_sequence: BaseExoEgoSequence = config.dataset.setup()  # one-liner
    rr.log("/", exoego_sequence.world_coordinate_system, static=True)
    set_annotation_context()

    parent_log_path = Path("world")
    timeline: str = "video_time"

    scene_setup_result: SceneSetupResult = setup_scene(exoego_sequence, parent_log_path, timeline)
    log_paths: LogPaths = scene_setup_result.log_paths
    shortest_timestamp: Int[ndarray, "n_frames"] = scene_setup_result.shortest_timestamp

    blueprint: rrb.Blueprint = create_blueprint(
        exo_video_log_paths=log_paths.exo_video_log_paths,
        ego_video_log_paths=log_paths.ego_video_log_paths,
    )
    rr.send_blueprint(blueprint)

    if config.log_labels:
        log_exoego_batch(
            exoego_sequence=exoego_sequence,
            timeline=timeline,
            shortest_timestamp=shortest_timestamp,
            parent_log_path=parent_log_path,
        )

    hand_detection_engine = HandDetector(HandDetectorConfig(verbose=False))
    hand_keypoint_engine = WilorHandKeypointDetector(HandKeypointDetectorConfig(verbose=False))

    exo_sequence: BaseExoSequence | None = exoego_sequence.exo_sequence
    if exo_sequence is None:
        raise ValueError("Selected dataset does not expose an exocentric camera rig.")

    target_ts_nano: int = (
        config.hand_calibrator.ts_nano
        if config.hand_calibrator.ts_nano is not None
        else frame_index_to_timestamp(shortest_timestamp, 0)
    )
    frame_index: int = timestamp_to_frame_index(target_ts_nano, shortest_timestamp)
    frame_timestamp_ns: int = frame_index_to_timestamp(shortest_timestamp, frame_index)
    frame_timestamp_seconds: float = frame_timestamp_ns * 1e-9
    rr.set_time(timeline=timeline, duration=frame_timestamp_seconds)

    rgb_ts_batch: UInt8[ndarray, "n_frames n_views H W 3"] = mv_reader_to_rgb_ts_batch(
        mv_reader=exo_sequence.exo_video_readers,
        num_frames=config.hand_calibrator.n_frame_optim,
        ts_nanos=target_ts_nano,
        frame_timestamps_ns=shortest_timestamp,
    )
    hand_calibrator = HandCalibrator(
        hand_detector=hand_detection_engine,
        hand_keypoint_detector=hand_keypoint_engine,
        config=config.hand_calibrator,
    )
    calibration_result: HandCalibrationResult = hand_calibrator(
        exo_cam_list=exo_sequence.exo_cam_list,
        rgb_ts_batch=rgb_ts_batch,
        parent_log_path=parent_log_path,
    )

    print(f"Total time taken: {timer() - start_time:.2f} seconds")
