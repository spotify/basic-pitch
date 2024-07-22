#!/usr/bin/env python
# encoding: utf-8
#
# Copyright 2024 Spotify AB
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a cos.pathy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and

import argparse
import logging

from basic_pitch.data import commandline
from basic_pitch.data.datasets.guitarset import main as guitarset_main
from basic_pitch.data.datasets.ikala import main as ikala_main
from basic_pitch.data.datasets.maestro import main as maestro_main
from basic_pitch.data.datasets.medleydb_pitch import main as medleydb_pitch_main
from basic_pitch.data.datasets.slakh import main as slakh_main

logger = logging.getLogger()
logger.setLevel(logging.INFO)

DATASET_DICT = {
    "guitarset": guitarset_main,
    "ikala": ikala_main,
    "maestro": maestro_main,
    "medleydb_pitch": medleydb_pitch_main,
    "slakh": slakh_main,
}


def main() -> None:
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
        logger.info(f"known_args:: {arg} = {getattr(known_args, arg)}")
    logger.info(f"pipeline_args = {pipeline_args}")
    DATASET_DICT[dataset](known_args, pipeline_args)


if __name__ == "__main__":
    main()
