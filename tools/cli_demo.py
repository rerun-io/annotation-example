import tyro

from annotation_example.api.process_data import ProcessConfig, process_data

if __name__ == "__main__":
    process_data(tyro.cli(ProcessConfig))
