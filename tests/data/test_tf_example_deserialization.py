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
import numpy as np
import pathlib
import shutil
import tensorflow as tf

from apache_beam.testing.test_pipeline import TestPipeline
from typing import List

from basic_pitch.constants import Split
from basic_pitch.data.datasets.guitarset import GuitarSetToTfExample
from basic_pitch.data.pipeline import WriteBatchToTfRecord
from basic_pitch.data.tf_example_deserialization import (
    prepare_datasets,
    prepare_visualization_datasets,
    sample_datasets,
    transcription_file_generator,
)

from utils import create_mock_wav

RESOURCES_PATH = pathlib.Path(__file__).parent.parent / "resources"
TRAIN_TRACK_ID = "00_BN1-129-Eb_comp"
VALID_TRACK_ID = "00_BN1-147-Gb_comp"


def create_empty_tfrecord(filepath: pathlib.Path) -> None:
    assert filepath.suffix == ".tfrecord"
    with tf.io.TFRecordWriter(str(filepath)) as writer:
        writer.write("")


def create_tfrecord(input_data: List[str], dataset_home: str, output_dir: str) -> None:
    with TestPipeline() as p:
        (
            p
            | "Create PCollection of track IDs" >> beam.Create([input_data])
            | "Create tf.Example" >> beam.ParDo(GuitarSetToTfExample(dataset_home, download=False))
            | "Write to tfrecord" >> beam.ParDo(WriteBatchToTfRecord(output_dir))
        )


def setup_test_resources(tmp_path: pathlib.Path) -> pathlib.Path:
    mock_guitarset_home = tmp_path / "guitarset"
    mock_guitarset_audio = mock_guitarset_home / "audio_mono-mic"
    mock_guitarset_annotations = mock_guitarset_home / "annotation"

    mock_guitarset_audio.mkdir(parents=True)
    mock_guitarset_annotations.mkdir(parents=True)

    output_home = tmp_path / "data" / "basic_pitch"
    output_splits_dir = output_home / "guitarset" / "splits"

    def mock_and_process(split: str, track_id: str) -> None:
        create_mock_wav(mock_guitarset_audio / f"{track_id}_mic.wav", duration_min=1)
        shutil.copy(
            RESOURCES_PATH / "data" / "guitarset" / "annotation" / f"{track_id}.jams",
            mock_guitarset_annotations / f"{track_id}.jams",
        )

        output_dir = output_splits_dir / split
        output_dir.mkdir(parents=True)

        create_tfrecord(input_data=[track_id], dataset_home=str(mock_guitarset_home), output_dir=str(output_dir))

    mock_and_process("train", TRAIN_TRACK_ID)
    mock_and_process("validation", VALID_TRACK_ID)

    return output_home


def test_prepare_datasets(tmp_path: pathlib.Path) -> None:
    datasets_home = setup_test_resources(tmp_path)

    ds_train, ds_valid = prepare_datasets(
        datasets_base_path=str(datasets_home),
        training_shuffle_buffer_size=1,
        batch_size=1,
        validation_steps=1,
        datasets_to_use=["guitarset"],
        dataset_sampling_frequency=np.array([1]),
    )

    assert ds_train is not None and isinstance(ds_train, tf.data.Dataset)
    assert ds_valid is not None and isinstance(ds_valid, tf.data.Dataset)


def test_prepare_visualization_dataset(tmp_path: pathlib.Path) -> None:
    datasets_home = setup_test_resources(tmp_path)

    ds_train, ds_valid = prepare_visualization_datasets(
        datasets_base_path=str(datasets_home),
        batch_size=1,
        validation_steps=1,
        datasets_to_use=["guitarset"],
        dataset_sampling_frequency=np.array([1]),
    )

    assert ds_train is not None and isinstance(ds_train, tf.data.Dataset)
    assert ds_valid is not None and isinstance(ds_train, tf.data.Dataset)


def test_sample_datasets(tmp_path: pathlib.Path) -> None:
    """touches the following methods:
    - transcription_dataset
        - parse_transcription_tfexample
        - is_not_bad_shape
        - sparse2dense
        - reduce_transcription_inputs
        - get_sample_weights
            - _infer_time_size
        - get_transcription_chunks
            - extract_random_window
                - extract_window
                    - trim_time
        - is_not_all_silent_annotations
        - to_transcription_training_input
    """
    datasets_home = setup_test_resources(tmp_path)

    ds = sample_datasets(
        split=Split.train,
        datasets_base_path=str(datasets_home),
        datasets=["guitarset"],
        dataset_sampling_frequency=np.array([1]),
        n_shuffle=1,
        n_samples_per_track=1,
        pairs=True,
    )

    assert ds is not None and isinstance(ds, tf.data.Dataset)


def test_transcription_file_generator_train(tmp_path: pathlib.Path) -> None:
    dataset_path = tmp_path / "test_ds" / "splits" / Split.train.name
    dataset_path.mkdir(parents=True)
    create_empty_tfrecord(dataset_path / "test.tfrecord")

    file_gen, random_seed = transcription_file_generator(
        Split.train, ["test_ds"], datasets_base_path=str(tmp_path), sample_weights=np.array([1])
    )

    assert random_seed is False

    generator = file_gen()
    assert next(generator).numpy().decode("utf-8") == str(dataset_path / "test.tfrecord")
    try:
        next(generator)
    except Exception as e:
        assert isinstance(e, StopIteration)


def test_transcription_file_generator_valid(tmp_path: pathlib.Path) -> None:
    dataset_path = tmp_path / "test_ds" / "splits" / Split.validation.name
    dataset_path.mkdir(parents=True)
    create_empty_tfrecord(dataset_path / "test.tfrecord")

    file_gen, random_seed = transcription_file_generator(
        Split.validation, ["test_ds"], datasets_base_path=str(tmp_path), sample_weights=np.array([1])
    )

    assert random_seed is True

    generator = file_gen()
    assert next(generator).numpy().decode("utf-8") == str(dataset_path / "test.tfrecord")
    try:
        next(generator)
    except Exception as e:
        assert isinstance(e, StopIteration)
