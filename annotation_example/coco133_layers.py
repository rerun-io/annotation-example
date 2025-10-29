from __future__ import annotations

from enum import IntEnum
from typing import Final


class Coco133AnnotationLayer(IntEnum):
    """Semantic layers used when logging COCO-133 keypoints."""

    GT = 0
    RAW_2D = 1
    TRACKED_2D = 2
    PROJECTED_2D = 3
    OPTIMIZED_2D = 4


COCO133_LAYER_LABELS: Final[dict[Coco133AnnotationLayer, str]] = {
    Coco133AnnotationLayer.GT: "coco133_gt",
    Coco133AnnotationLayer.RAW_2D: "coco133_raw2d",
    Coco133AnnotationLayer.TRACKED_2D: "coco133_tracked2d",
    Coco133AnnotationLayer.PROJECTED_2D: "coco133_projected2d",
    Coco133AnnotationLayer.OPTIMIZED_2D: "coco133_optimized2d",
}

# Skeleton link colours should avoid the red/yellow/green spectrum that encodes per-keypoint confidence.
COCO133_LAYER_COLORS: Final[dict[Coco133AnnotationLayer, tuple[int, int, int]]] = {
    Coco133AnnotationLayer.GT: (30, 64, 255),  # deep blue for annotated ground truth
    Coco133AnnotationLayer.RAW_2D: (59, 130, 246),  # azure
    Coco133AnnotationLayer.TRACKED_2D: (165, 105, 255),  # violet
    Coco133AnnotationLayer.PROJECTED_2D: (217, 70, 239),  # magenta
    Coco133AnnotationLayer.OPTIMIZED_2D: (148, 163, 255),  # periwinkle
}

COCO133_PREDICTION_LAYER_TO_PATH: Final[dict[Coco133AnnotationLayer, str]] = {
    Coco133AnnotationLayer.RAW_2D: "raw",
    Coco133AnnotationLayer.TRACKED_2D: "tracked",
    Coco133AnnotationLayer.PROJECTED_2D: "projected",
    Coco133AnnotationLayer.OPTIMIZED_2D: "optimized",
}
