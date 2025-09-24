import tyro

from annotation_example.api.benchmark_hand_calib import BenchmarkHandCalibConfig, main

if __name__ == "__main__":
    main(tyro.cli(BenchmarkHandCalibConfig))
