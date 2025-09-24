from dataclasses import dataclass
from pathlib import Path
from timeit import default_timer as timer

import rerun as rr
import rerun.blueprint as rrb
from jaxtyping import Int
from numpy import ndarray
from simplecv.apis.view_exoego import (
    LogPaths,
    SceneSetupResult,
    create_blueprint,
    log_exoego_batch,
    set_annotation_context,
    setup_scene,
)
from simplecv.configs.exoego_dataset_configs import AnnotatedExoEgoDatasetUnion
from simplecv.data.exoego.base_exoego import BaseExoEgoSequence
from simplecv.rerun_log_utils import (
    RerunTyroConfig,
)


class HandCalibrator:
    def __init__(self):
        pass

    def __call__(self):
        pass


@dataclass
class BenchmarkHandCalibConfig:
    rr_config: RerunTyroConfig
    dataset: AnnotatedExoEgoDatasetUnion


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

    log_exoego_batch(
        exoego_sequence=exoego_sequence,
        timeline=timeline,
        shortest_timestamp=shortest_timestamp,
        parent_log_path=parent_log_path,
    )

    print(f"Total time taken: {timer() - start_time:.2f} seconds")
