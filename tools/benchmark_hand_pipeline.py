import tyro

from annotation_example.api.benchmark_hand_pipeline import BenchmarkHandCalibConfig, main

if __name__ == "__main__":
    main(tyro.cli(BenchmarkHandCalibConfig))
