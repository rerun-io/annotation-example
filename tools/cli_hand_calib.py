import tyro

from annotation_example.api.calibrate_hand import HandCalibConfig, main

if __name__ == "__main__":
    main(tyro.cli(HandCalibConfig))
