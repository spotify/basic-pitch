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
import logging
import numpy as np
import os
import pathlib
from typing import List
import wave

import apache_beam as beam
from apache_beam.testing.test_pipeline import TestPipeline

from basic_pitch.data.datasets.maestro import (
    MaestroToTfExample,
    MaestroInvalidTracks,
    create_input_data,
)
from basic_pitch.data.pipeline import WriteBatchToTfRecord

RESOURCES_PATH = pathlib.Path(__file__).parent.parent / "resources"
MAESTRO_TEST_DATA_PATH = RESOURCES_PATH / "data" / "maestro"

TRAIN_TRACK_ID = "2004/MIDI-Unprocessed_SMF_05_R1_2004_01_ORIG_MID--AUDIO_05_R1_2004_03_Track03_wav"
VALID_TRACK_ID = "2004/MIDI-Unprocessed_SMF_02_R1_2004_01-05_ORIG_MID--AUDIO_02_R1_2004_06_Track06_wav"
TEST_TRACK_ID = "2004/MIDI-Unprocessed_SMF_02_R1_2004_01-05_ORIG_MID--AUDIO_02_R1_2004_08_Track08_wav"

MOCK_15M_TRACK_ID = "2004/MIDI-Unprocessed_SMF_02_R1_2004_01-05_ORIG_MID--AUDIO_02_R1_2004_05_Track05_wav"


def create_mock_wav(output_fpath: str, duration_min: int) -> None:
    duration_seconds = duration_min * 60
    sample_rate = 44100
    n_channels = 2  # Stereo
    sampwidth = 2  # 2 bytes per sample (16-bit audio)

    # Generate a silent audio data array
    num_samples = duration_seconds * sample_rate
    audio_data = np.zeros((num_samples, n_channels), dtype=np.int16)

    # Create the WAV file
    with wave.open(str(output_fpath), "w") as wav_file:
        wav_file.setnchannels(n_channels)
        wav_file.setsampwidth(sampwidth)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(audio_data.tobytes())

    logging.info(f"Mock {duration_min}-minute WAV file '{output_fpath}' created successfully.")


def test_maestro_to_tf_example(tmpdir: str) -> None:
    input_data: List[str] = [TRAIN_TRACK_ID]
    with TestPipeline() as p:
        (
            p
            | "Create PCollection of track IDs" >> beam.Create([input_data])
            | "Create tf.Example" >> beam.ParDo(MaestroToTfExample(str(MAESTRO_TEST_DATA_PATH), download=False))
            | "Write to tfrecord" >> beam.ParDo(WriteBatchToTfRecord(tmpdir))
        )

    assert len(os.listdir(tmpdir)) == 1
    assert os.path.splitext(os.listdir(tmpdir)[0])[-1] == ".tfrecord"
    with open(os.path.join(tmpdir, os.listdir(tmpdir)[0]), "rb") as fp:
        data = fp.read()
        assert len(data) != 0


def test_maestro_invalid_tracks(tmpdir: str) -> None:
    input_data = [(TRAIN_TRACK_ID, "train"), (VALID_TRACK_ID, "validation"), (TEST_TRACK_ID, "test")]
    split_labels = set([e[1] for e in input_data])
    with TestPipeline() as p:
        splits = (
            p
            | "Create PCollection" >> beam.Create(input_data)
            | "Tag it" >> beam.ParDo(MaestroInvalidTracks(str(MAESTRO_TEST_DATA_PATH))).with_outputs(*split_labels)
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


def test_maestro_invalid_tracks_over_15_min(tmpdir: str) -> None:
    """
    The track id used here is a real track id in maestro, and it is part of the train split, but we mock the data so as
    not to store a large file in git, hence the variable name.
    """

    mock_fpath = MAESTRO_TEST_DATA_PATH / "2004" / (MOCK_15M_TRACK_ID.split("/")[1] + ".wav")
    create_mock_wav(str(mock_fpath), 16)

    input_data = [(MOCK_15M_TRACK_ID, "train")]
    split_labels = set([e[1] for e in input_data])
    with TestPipeline() as p:
        splits = (
            p
            | "Create PCollection" >> beam.Create(input_data)
            | "Tag it" >> beam.ParDo(MaestroInvalidTracks(str(MAESTRO_TEST_DATA_PATH))).with_outputs(*split_labels)
        )

        for split in split_labels:
            (
                getattr(splits, split)
                | f"Write {split} to text"
                >> beam.io.WriteToText(os.path.join(tmpdir, f"output_{split}.txt"), shard_name_template="")
            )

    for _, split in input_data:
        with open(os.path.join(tmpdir, f"output_{split}.txt"), "r") as fp:
            assert fp.read().strip() == ""

    os.remove(mock_fpath)


def test_maestro_create_input_data() -> None:
    data = create_input_data(str(MAESTRO_TEST_DATA_PATH))
    assert len(data)

    splits = set(map(lambda el: el[1], data))
    assert splits == set(["train", "validation", "test"])
