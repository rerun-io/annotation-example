import tyro

from annotation_example.api.full_exoego_pipeline import RRDPipelineConfig, main

if __name__ == "__main__":
    main(tyro.cli(RRDPipelineConfig))
