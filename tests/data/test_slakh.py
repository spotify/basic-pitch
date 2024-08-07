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
import shutil

from typing import List, Tuple

from apache_beam.testing.test_pipeline import TestPipeline

from basic_pitch.data.datasets.slakh import (
    SlakhFilterInvalidTracks,
    SlakhToTfExample,
    create_input_data,
)
from basic_pitch.data.pipeline import WriteBatchToTfRecord

from utils import create_mock_flac, create_mock_midi

RESOURCES_PATH = pathlib.Path(__file__).parent.parent / "resources"
SLAKH_PATH = RESOURCES_PATH / "data" / "slakh" / "slakh2100_flac_redux"

TRAIN_PIANO_TRACK_ID = "Track00001-S02"
TRAIN_DRUMS_TRACK_ID = "Track00001-S01"

VALID_PIANO_TRACK_ID = "Track01501-S06"
VALID_DRUMS_TRACK_ID = "Track01501-S03"

TEST_PIANO_TRACK_ID = "Track01876-S01"
TEST_DRUMS_TRACK_ID = "Track01876-S08"

OMITTED_PIANO_TRACK_ID = "Track00049-S05"
OMITTED_DRUMS_TRACK_ID = "Track00049-S06"


# Function to generate a sine wave
def create_mock_input_data(data_home: pathlib.Path, input_data: List[Tuple[str, str]]) -> None:
    for track_id, split in input_data:
        track_num, inst_num = track_id.split("-")
        track_dir = data_home / split / track_num

        stems_dir = track_dir / "stems"
        stems_dir.mkdir(parents=True, exist_ok=True)
        create_mock_flac(stems_dir / (inst_num + ".flac"))

        midi_dir = track_dir / "MIDI"
        midi_dir.mkdir(parents=True, exist_ok=True)
        create_mock_midi(midi_dir / (inst_num + ".mid"))

        shutil.copy(SLAKH_PATH / split / track_num / "metadata.yaml", track_dir / "metadata.yaml")


def test_slakh_to_tf_example(tmp_path: pathlib.Path) -> None:
    mock_slakh_home = tmp_path / "slakh"
    mock_slakh_ext = mock_slakh_home / "slakh2100_flac_redux"

    input_data: List[Tuple[str, str]] = [(TRAIN_PIANO_TRACK_ID, "train")]
    create_mock_input_data(mock_slakh_ext, input_data)

    output_dir = tmp_path / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)

    with TestPipeline() as p:
        (
            p
            | "Create PCollection of track IDs" >> beam.Create([[track_id for track_id, _ in input_data]])
            | "Create tf.Example" >> beam.ParDo(SlakhToTfExample(str(mock_slakh_home), download=False))
            | "Write to tfrecord" >> beam.ParDo(WriteBatchToTfRecord(str(output_dir)))
        )

    listdir = os.listdir(output_dir)
    assert len(listdir) == 1
    assert os.path.splitext(listdir[0])[-1] == ".tfrecord"
    with open(output_dir / listdir[0], "rb") as fp:
        data = fp.read()
        assert len(data) != 0


def test_slakh_invalid_tracks(tmp_path: pathlib.Path) -> None:
    mock_slakh_home = tmp_path / "slakh"
    mock_slakh_ext = mock_slakh_home / "slakh2100_flac_redux"

    split_labels = ["train", "validation", "test"]
    input_data = [(TRAIN_PIANO_TRACK_ID, "train"), (VALID_PIANO_TRACK_ID, "validation"), (TEST_PIANO_TRACK_ID, "test")]
    create_mock_input_data(mock_slakh_ext, input_data)

    with TestPipeline() as p:
        splits = (
            p
            | "Create PCollection" >> beam.Create(input_data)
            | "Tag it" >> beam.ParDo(SlakhFilterInvalidTracks(str(mock_slakh_home))).with_outputs(*split_labels)
        )

        for split in split_labels:
            (
                getattr(splits, split)
                | f"Write {split} to text"
                >> beam.io.WriteToText(str(tmp_path / f"output_{split}.txt"), shard_name_template="")
            )

    for track_id, split in input_data:
        with open(tmp_path / f"output_{split}.txt", "r") as fp:
            assert fp.read().strip() == track_id


def test_slakh_invalid_tracks_omitted(tmp_path: pathlib.Path) -> None:
    mock_slakh_home = tmp_path / "slakh"
    mock_slakh_ext = mock_slakh_home / "slakh2100_flac_redux"

    split_labels = ["train", "omitted"]
    input_data = [(TRAIN_PIANO_TRACK_ID, "train"), (OMITTED_PIANO_TRACK_ID, "omitted")]
    create_mock_input_data(mock_slakh_ext, input_data)

    with TestPipeline() as p:
        splits = (
            p
            | "Create PCollection" >> beam.Create(input_data)
            | "Tag it" >> beam.ParDo(SlakhFilterInvalidTracks(str(mock_slakh_home))).with_outputs(*split_labels)
        )

        for split in split_labels:
            (
                getattr(splits, split)
                | f"Write {split} to text"
                >> beam.io.WriteToText(str(tmp_path / f"output_{split}.txt"), shard_name_template="")
            )

    with open(tmp_path / "output_train.txt", "r") as fp:
        assert fp.read().strip() == TRAIN_PIANO_TRACK_ID

    with open(tmp_path / "output_omitted.txt", "r") as fp:
        assert fp.read().strip() == ""


def test_slakh_invalid_tracks_drums(tmp_path: pathlib.Path) -> None:
    mock_slakh_home = tmp_path / "slakh"
    mock_slakh_ext = mock_slakh_home / "slakh2100_flac_redux"

    split_labels = ["train", "validation", "test"]
    input_data = [(TRAIN_DRUMS_TRACK_ID, "train"), (VALID_DRUMS_TRACK_ID, "validation"), (TEST_DRUMS_TRACK_ID, "test")]
    create_mock_input_data(mock_slakh_ext, input_data)

    with TestPipeline() as p:
        splits = (
            p
            | "Create PCollection" >> beam.Create(input_data)
            | "Tag it" >> beam.ParDo(SlakhFilterInvalidTracks(str(mock_slakh_home))).with_outputs(*split_labels)
        )

        for split in split_labels:
            (
                getattr(splits, split)
                | f"Write {split} to text"
                >> beam.io.WriteToText(str(tmp_path / f"output_{split}.txt"), shard_name_template="")
            )

    for _, split in input_data:
        with open(tmp_path / f"output_{split}.txt", "r") as fp:
            assert fp.read().strip() == ""


def test_create_input_data() -> None:
    data = create_input_data()
    for _, group in itertools.groupby(data, lambda el: el[1]):
        assert len(list(group))
