import tyro

from annotation_example.api.calibrate_mv_videos import VGGTInferenceConfig, main

if __name__ == "__main__":
    main(tyro.cli(VGGTInferenceConfig))
