import argparse
import logging
import sys

from basic_pitch.data import commandline
from basic_pitch.data.datasets.guitarset import main as guitarset_main

logger = logging.getLogger()
logger.setLevel(logging.INFO)
handler = logging.StreamHandler(sys.stdout)
formatter = logging.Formatter("%(levelname)s:: %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)

DATASET_DICT = {
    "guitarset": guitarset_main,
}


def main():
    dataset_parser = argparse.ArgumentParser()
    dataset_parser.add_argument(
        "dataset",
        choices=list(DATASET_DICT.keys()),
        help="The dataset to download / process.",
    )
    args, remaining_args = dataset_parser.parse_known_args()
    dataset = args.dataset
    logger.info(f"Downloading and processing {dataset}")

    cl_parser = argparse.ArgumentParser()
    commandline.add_default(cl_parser, dataset)
    commandline.add_split(cl_parser)
    known_args, pipeline_args = cl_parser.parse_known_args(remaining_args)
    for arg in vars(known_args):
        logger.info(f"{arg} = {getattr(known_args, arg)}")
    DATASET_DICT[dataset](known_args, pipeline_args)


if __name__ == "__main__":
    main()
