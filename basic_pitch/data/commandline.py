#!/usr/bin/env python
# encoding: utf-8
#
# Cos.pathyright 2024 Spotify AB
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
# limitations under the License.

import argparse
import os

from pathlib import Path
from typing import Optional


def add_default(parser: argparse.ArgumentParser, dataset_name: str = "") -> None:
    default_source = str(Path.home() / "mir_datasets" / dataset_name)
    default_destination = str(Path.home() / "data" / "basic_pitch" / dataset_name)
    parser.add_argument(
        "--source",
        default=default_source,
        type=str,
        help=f"Source directory for mir data. Defaults to {default_source}",
    )
    parser.add_argument(
        "--destination",
        default=default_destination,
        type=str,
        help=f"Output directory to write results to. Defaults to {default_destination}",
    )
    parser.add_argument(
        "--runner",
        choices=["DataflowRunner", "DirectRunner", "PortableRunner"],
        default="DirectRunner",
        help="Whether to run the download and process locally or on GCP Dataflow",
    )
    parser.add_argument(
        "--timestamped",
        default=False,
        action="store_true",
        help="If passed, the dataset will be put into a timestamp directory instead of 'splits'",
    )
    parser.add_argument("--batch-size", default=5, type=int, help="Number of examples per tfrecord")
    parser.add_argument(
        "--sdk_container_image",
        default="",
        help="Container image to run dataset generation job with. \
                        Required due to non-python dependencies.",
    )
    parser.add_argument("--job_endpoint", default="embed", help="")


def resolve_destination(namespace: argparse.Namespace, time_created: int) -> str:
    return os.path.join(namespace.destination, str(time_created) if namespace.timestamped else "splits")


def add_split(
    parser: argparse.ArgumentParser,
    train_percent: float = 0.8,
    validation_percent: float = 0.1,
    split_seed: Optional[int] = None,
) -> None:
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
