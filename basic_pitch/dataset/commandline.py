#!/usr/bin/env python
# encoding: utf-8
#
# Copyright 2022 Spotify AB
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import os


def add_default(parser: argparse.ArgumentParser):
    parser.add_argument("--runner", choices=["DataflowRunner", "DirectRunner"], default="DirectRunner")
    parser.add_argument(
        "source",
        help="Source directory for mir data.",
    )
    parser.add_argument(
        "destination",
        help="Output directory to write results to.",
    )
    parser.add_argument(
        "--timestamped",
        default=False,
        action="store_true",
        help="If passed, the dataset will be put into a timestamp directory instead of 'splits'",
    )
    parser.add_argument("--batch-size", default=5, type=int, help="Number of examples per tfrecord")
    parser.add_argument(
        "--worker-harness-container-image",
        default="",
        help="Container image to run dataset generation job with. Required due to non-python dependencies",
    )


def resolve_destination(namespace: argparse.Namespace, dataset: str, time_created: int) -> str:
    return os.path.join(namespace.destination, dataset, str(time_created) if namespace.timestamped else "splits")


def add_split(
    parser: argparse.ArgumentParser,
    train_percent: float = 0.8,
    validation_percent: float = 0.1,
    split_seed: int = None,
):
    parser.add_argument(
        "--train-percent",
        type=float,
        default=train_percent,
        help="Percentage of tracks to mark as train",
    )
    parser.add_argument(
        "--validation-percent",
        type=float,
        default=validation_percent,
        help="Percentage of tracks to mark as validation",
    )
    parser.add_argument(
        "--split-seed",
        type=int,
        default=split_seed,
        help="Seed for random number generator used in split generation",
    )
