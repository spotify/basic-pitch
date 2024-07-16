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
import pathlib

from typing import List

from apache_beam.testing.test_pipeline import TestPipeline

from basic_pitch.data.datasets.slakh import (
    SlakhFilterInvalidTracks,
    SlakhToTfExample,
    create_input_data,
)
from basic_pitch.data.pipeline import WriteBatchToTfRecord

RESOURCES_PATH = pathlib.Path(__file__).parent.parent / "resources"
TRAIN_PIANO_TRACK_ID = "Track00001-S02"
TRAIN_DRUMS_TRACK_ID = "Track00001-S01"
VALID_PIANO_TRACK_ID = "Track01501-S06"
VALID_DRUMS_TRACK_ID = "Track01501-S03"
TEST_PIANO_TRACK_ID = "Track01876-S01"
TEST_DRUMS_TRACK_ID = "Track01876-S08"
OMITTED_PIANO_TRACK_ID = "Track00049-S05"
OMITTED_DRUMS_TRACK_ID = "Track00049-S06"


def test_slakh_to_tf_example(tmpdir: str) -> None:
    input_data: List[str] = [TRAIN_PIANO_TRACK_ID]
    with TestPipeline() as p:
        (
            p
            | "Create PCollection of track IDs" >> beam.Create([input_data])
            | "Create tf.Example"
            >> beam.ParDo(SlakhToTfExample(str(RESOURCES_PATH / "data" / "slakh"), download=False))
            | "Write to tfrecord" >> beam.ParDo(WriteBatchToTfRecord(tmpdir))
        )

    assert len(os.listdir(tmpdir)) == 1
    assert os.path.splitext(os.listdir(tmpdir)[0])[-1] == ".tfrecord"
    with open(os.path.join(tmpdir, os.listdir(tmpdir)[0]), "rb") as fp:
        data = fp.read()
        assert len(data) != 0


def test_slakh_invalid_tracks(tmpdir: str) -> None:
    split_labels = ["train", "validation", "test"]
    input_data = [(TRAIN_PIANO_TRACK_ID, "train"), (VALID_PIANO_TRACK_ID, "validation"), (TEST_PIANO_TRACK_ID, "test")]

    with TestPipeline() as p:
        splits = (
            p
            | "Create PCollection" >> beam.Create(input_data)
            | "Tag it"
            >> beam.ParDo(SlakhFilterInvalidTracks(str(RESOURCES_PATH / "data" / "slakh"))).with_outputs(*split_labels)
        )

        for split in split_labels:
            (
                getattr(splits, split)
                | f"Write {split} to text"
                >> beam.io.WriteToText(os.path.join(tmpdir, f"output_{split}.txt"), shard_name_template="")
            )

    for track_id, split in input_data:
        with open(os.path.join(tmpdir, f"output_{split}.txt"), "r") as fp:
            assert fp.read().strip() == track_id


def test_slakh_invalid_tracks_omitted(tmpdir: str) -> None:
    split_labels = ["train", "omitted"]
    input_data = [(TRAIN_PIANO_TRACK_ID, "train"), (OMITTED_PIANO_TRACK_ID, "omitted")]

    with TestPipeline() as p:
        splits = (
            p
            | "Create PCollection" >> beam.Create(input_data)
            | "Tag it"
            >> beam.ParDo(SlakhFilterInvalidTracks(str(RESOURCES_PATH / "data" / "slakh"))).with_outputs(*split_labels)
        )

        for split in split_labels:
            (
                getattr(splits, split)
                | f"Write {split} to text"
                >> beam.io.WriteToText(os.path.join(tmpdir, f"output_{split}.txt"), shard_name_template="")
            )

    with open(os.path.join(tmpdir, "output_train.txt"), "r") as fp:
        assert fp.read().strip() == TRAIN_PIANO_TRACK_ID

    with open(os.path.join(tmpdir, "output_omitted.txt"), "r") as fp:
        assert fp.read().strip() == ""


def test_slakh_invalid_tracks_drums(tmpdir: str) -> None:
    split_labels = ["train", "validation", "test"]
    input_data = [(TRAIN_DRUMS_TRACK_ID, "train"), (VALID_DRUMS_TRACK_ID, "validation"), (TEST_DRUMS_TRACK_ID, "test")]

    with TestPipeline() as p:
        splits = (
            p
            | "Create PCollection" >> beam.Create(input_data)
            | "Tag it"
            >> beam.ParDo(SlakhFilterInvalidTracks(str(RESOURCES_PATH / "data" / "slakh"))).with_outputs(*split_labels)
        )

        for split in split_labels:
            (
                getattr(splits, split)
                | f"Write {split} to text"
                >> beam.io.WriteToText(os.path.join(tmpdir, f"output_{split}.txt"), shard_name_template="")
            )

    for track_id, split in input_data:
        with open(os.path.join(tmpdir, f"output_{split}.txt"), "r") as fp:
            assert fp.read().strip() == ""


def test_create_input_data() -> None:
    data = create_input_data()
    for key, group in itertools.groupby(data, lambda el: el[1]):
        assert len(list(group))
