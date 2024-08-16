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
import shutil

from apache_beam.testing.test_pipeline import TestPipeline
from typing import List

from basic_pitch.data.datasets.guitarset import (
    GuitarSetToTfExample,
    GuitarSetInvalidTracks,
    create_input_data,
)
from basic_pitch.data.pipeline import WriteBatchToTfRecord

from utils import create_mock_wav

RESOURCES_PATH = pathlib.Path(__file__).parent.parent / "resources"
TRACK_ID = "00_BN1-129-Eb_comp"


def test_guitarset_to_tf_example(tmp_path: pathlib.Path) -> None:
    mock_guitarset_home = tmp_path / "guitarset"
    mock_guitarset_audio = mock_guitarset_home / "audio_mono-mic"
    mock_guitarset_annotations = mock_guitarset_home / "annotation"
    output_dir = tmp_path / "output"

    mock_guitarset_audio.mkdir(parents=True)
    mock_guitarset_annotations.mkdir(parents=True)
    output_dir.mkdir()

    create_mock_wav(mock_guitarset_audio / f"{TRACK_ID}_mic.wav", duration_min=1)
    shutil.copy(
        RESOURCES_PATH / "data" / "guitarset" / "annotation" / f"{TRACK_ID}.jams",
        mock_guitarset_annotations / f"{TRACK_ID}.jams",
    )

    input_data: List[str] = [TRACK_ID]
    with TestPipeline() as p:
        (
            p
            | "Create PCollection of track IDs" >> beam.Create([input_data])
            | "Create tf.Example" >> beam.ParDo(GuitarSetToTfExample(str(mock_guitarset_home), download=False))
            | "Write to tfrecord" >> beam.ParDo(WriteBatchToTfRecord(str(output_dir)))
        )

    listdir = os.listdir(output_dir)
    assert len(listdir) == 1
    assert os.path.splitext(listdir[0])[-1] == ".tfrecord"
    with open(output_dir / listdir[0], "rb") as fp:
        data = fp.read()
        assert len(data) != 0


def test_guitarset_invalid_tracks(tmpdir: str) -> None:
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


def test_guitarset_create_input_data() -> None:
    data = create_input_data(train_percent=0.33, validation_percent=0.33)
    data.sort(key=lambda el: el[1])  # sort by split
    tolerance = 0.1
    for _, group in itertools.groupby(data, lambda el: el[1]):
        assert (0.33 - tolerance) * len(data) <= len(list(group)) <= (0.33 + tolerance) * len(data)


def test_guitarset_create_input_data_overallocate() -> None:
    try:
        create_input_data(train_percent=0.6, validation_percent=0.6)
    except AssertionError:
        assert True
    else:
        assert False
