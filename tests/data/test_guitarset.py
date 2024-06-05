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
import apache_beam as beam
import itertools
import os
import pathlib

from apache_beam.testing.test_pipeline import TestPipeline
from typing import List

from basic_pitch.data.datasets.guitarset import (
    GuitarSetToTfExample,
    GuitarSetInvalidTracks,
    create_input_data,
)
from basic_pitch.data.pipeline import WriteBatchToTfRecord

RESOURCES_PATH = pathlib.Path(__file__).parent.parent / "resources"
TRACK_ID = "00_BN1-129-Eb_comp"


def test_guitar_set_to_tf_example(tmpdir: str) -> None:
    input_data: List[str] = [TRACK_ID]
    with TestPipeline() as p:
        (
            p
            | "Create PCollection of track IDs" >> beam.Create([input_data])
            | "Create tf.Example"
            >> beam.ParDo(GuitarSetToTfExample(str(RESOURCES_PATH / "data" / "guitarset"), download=False))
            | "Write to tfrecord" >> beam.ParDo(WriteBatchToTfRecord(tmpdir))
        )

    assert len(os.listdir(tmpdir)) == 1
    assert os.path.splitext(os.listdir(tmpdir)[0])[-1] == ".tfrecord"
    with open(os.path.join(tmpdir, os.listdir(tmpdir)[0]), "rb") as fp:
        data = fp.read()
        assert len(data) != 0


def test_guitar_set_invalid_tracks(tmpdir: str) -> None:
    split_labels = ["train", "test", "validation"]
    input_data = [(str(i), split) for i, split in enumerate(split_labels)]
    with TestPipeline() as p:
        splits = (
            p
            | "Create PCollection" >> beam.Create(input_data)
            | "Tag it" >> beam.ParDo(GuitarSetInvalidTracks()).with_outputs(*split_labels)
        )

        for split in split_labels:
            (
                getattr(splits, split)
                | f"Write {split} to text"
                >> beam.io.WriteToText(os.path.join(tmpdir, f"output_{split}.txt"), shard_name_template="")
            )

    for i, split in enumerate(split_labels):
        with open(os.path.join(tmpdir, f"output_{split}.txt"), "r") as fp:
            assert fp.read().strip() == str(i)


def test_create_input_data() -> None:
    data = create_input_data(train_percent=0.33, validation_percent=0.33)
    data.sort(key=lambda el: el[1])  # sort by split
    tolerance = 0.1
    for key, group in itertools.groupby(data, lambda el: el[1]):
        assert (0.33 - tolerance) * len(data) <= len(list(group)) <= (0.33 + tolerance) * len(data)


def test_create_input_data_overallocate() -> None:
    try:
        create_input_data(train_percent=0.6, validation_percent=0.6)
    except AssertionError:
        assert True
    else:
        assert False
