import tyro

from annotation_example.api.calibrate_mv_videos import CalibrateConfig, calibrate_mv_videos

if __name__ == "__main__":
    calibrate_mv_videos(tyro.cli(CalibrateConfig))
