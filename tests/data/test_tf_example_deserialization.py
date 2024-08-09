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

import numpy as np
import os
import pathlib
import tensorflow as tf


from basic_pitch.data.tf_example_deserialization import transcription_dataset, transcription_file_generator


def create_empty_tfrecord(filepath: pathlib.Path) -> None:
    assert filepath.suffix == ".tfrecord"
    with tf.io.TFRecordWriter(str(filepath)) as writer:
        writer.write("")


# def test_prepare_dataset() -> None:
#     pass


# def test_sample_datasets() -> None:
#     pass


# def test_transcription_dataset(tmp_path: pathlib.Path) -> None:
#     dataset_path = tmp_path / "test_ds" / "splits" / "train"
#     dataset_path.mkdir(parents=True)
#     create_empty_tfrecord(dataset_path / "test.tfrecord")

#     file_gen, random_seed = transcription_file_generator(
#         "train", ["test_ds"], datasets_base_path=str(tmp_path), sample_weights=np.array([1])
#     )

#     transcription_dataset(file_generator=file_gen, n_samples_per_track=1, random_seed=random_seed)


def test_transcription_file_generator_train(tmp_path: pathlib.Path) -> None:
    dataset_path = tmp_path / "test_ds" / "splits" / "train"
    dataset_path.mkdir(parents=True)
    create_empty_tfrecord(dataset_path / "test.tfrecord")

    file_gen, random_seed = transcription_file_generator(
        "train", ["test_ds"], datasets_base_path=str(tmp_path), sample_weights=np.array([1])
    )

    assert random_seed is False

    generator = file_gen()
    assert next(generator).numpy().decode("utf-8") == str(dataset_path / "test.tfrecord")
    try:
        next(generator)
    except Exception as e:
        assert isinstance(e, StopIteration)


def test_transcription_file_generator_valid(tmp_path: pathlib.Path) -> None:
    dataset_path = tmp_path / "test_ds" / "splits" / "valid"
    dataset_path.mkdir(parents=True)
    create_empty_tfrecord(dataset_path / "test.tfrecord")

    file_gen, random_seed = transcription_file_generator(
        "valid", ["test_ds"], datasets_base_path=str(tmp_path), sample_weights=np.array([1])
    )

    assert random_seed is True

    generator = file_gen()
    assert next(generator).numpy().decode("utf-8") == str(dataset_path / "test.tfrecord")
    try:
        next(generator)
    except Exception as e:
        assert isinstance(e, StopIteration)
