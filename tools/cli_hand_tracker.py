import tyro

from annotation_example.api.hand_tracker import HandTrackingConfig, main

if __name__ == "__main__":
    main(tyro.cli(HandTrackingConfig))
