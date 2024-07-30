#!/usr/bin/env python
# encoding: utf-8
#
# Copyright 2024 Spotify AB
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

from apache_beam.testing.test_pipeline import TestPipeline

from basic_pitch.data.datasets.medleydb_pitch import (
    MedleyDbPitchInvalidTracks,
    create_input_data,
)


# TODO: Create test_medleydb_pitch_to_tf_example


def test_medleydb_pitch_invalid_tracks(tmpdir: str) -> None:
    split_labels = ["train", "validation"]
    input_data = [(str(i), split) for i, split in enumerate(split_labels)]
    with TestPipeline() as p:
        splits = (
            p
            | "Create PCollection" >> beam.Create(input_data)
            | "Tag it" >> beam.ParDo(MedleyDbPitchInvalidTracks()).with_outputs(*split_labels)
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


def test_medleydb_create_input_data() -> None:
    data = create_input_data(train_percent=0.5)
    data.sort(key=lambda el: el[1])  # sort by split
    tolerance = 0.01
    for _, group in itertools.groupby(data, lambda el: el[1]):
        assert (0.5 - tolerance) * len(data) <= len(list(group)) <= (0.5 + tolerance) * len(data)


def test_create_input_data_overallocate() -> None:
    try:
        create_input_data(train_percent=1.1)
    except AssertionError:
        assert True
    else:
        assert False
