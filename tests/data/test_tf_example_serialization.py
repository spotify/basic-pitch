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

import pathlib
import sox
import tensorflow as tf

from basic_pitch.constants import AUDIO_N_CHANNELS, AUDIO_SAMPLE_RATE
from basic_pitch.data.tf_example_serialization import to_transcription_tfexample

RESOURCES_PATH = pathlib.Path(__file__).parent.parent / "resources"


def test_to_transcription_tfexample(tmpdir: str) -> None:
    file_id = "test"
    source = "maestro"
    tmpfile = str(pathlib.Path(tmpdir) / "test.wav")
    notes_indices = [(0, 1), (1, 2)]
    notes_values = [30.0, 31.0]
    onsets_indices = [1, 2]
    onsets_values = [0.1, 0.2]
    contours_indices = [1, 2]
    contours_values = [0.1, 0.2]
    notes_onsets_shape = (1, 1)
    contours_shape = (1, 1)

    tfm = sox.Transformer()
    tfm.rate(AUDIO_SAMPLE_RATE)
    tfm.channels(AUDIO_N_CHANNELS)
    tfm.build(str(RESOURCES_PATH / "vocadito_10.wav"), tmpfile)
    example = to_transcription_tfexample(
        file_id=file_id,
        source=source,
        audio_wav_file_path=tmpfile,
        notes_indices=notes_indices,
        notes_values=notes_values,
        onsets_indices=onsets_indices,
        onsets_values=onsets_values,
        contours_indices=contours_indices,
        contours_values=contours_values,
        notes_onsets_shape=notes_onsets_shape,
        contours_shape=contours_shape,
    )
    assert type(example) is tf.train.Example
    assert example.features.feature["file_id"].bytes_list.value[0].decode("utf-8") == file_id
    assert example.features.feature["source"].bytes_list.value[0].decode("utf-8") == source
    assert example.features.feature["audio_wav"].bytes_list.value[0] == open(tmpfile, "rb").read()
    assert tf.reduce_all(
        tf.io.parse_tensor(example.features.feature["notes_indices"].bytes_list.value[0], out_type=tf.int64)
        == notes_indices
    )
    assert tf.reduce_all(
        tf.io.parse_tensor(example.features.feature["notes_values"].bytes_list.value[0], out_type=tf.float32)
        == notes_values
    )
    assert tf.reduce_all(
        tf.io.parse_tensor(example.features.feature["onsets_indices"].bytes_list.value[0], out_type=tf.int64)
        == onsets_indices
    )
    assert tf.reduce_all(
        tf.io.parse_tensor(example.features.feature["onsets_values"].bytes_list.value[0], out_type=tf.float32)
        == onsets_values
    )
    assert tf.reduce_all(
        tf.io.parse_tensor(example.features.feature["contours_indices"].bytes_list.value[0], out_type=tf.int64)
        == contours_indices
    )
    assert tf.reduce_all(
        tf.io.parse_tensor(example.features.feature["contours_values"].bytes_list.value[0], out_type=tf.float32)
        == contours_values
    )
    assert tf.reduce_all(
        tf.io.parse_tensor(example.features.feature["notes_onsets_shape"].bytes_list.value[0], out_type=tf.int64)
        == notes_onsets_shape
    )
    assert tf.reduce_all(
        tf.io.parse_tensor(example.features.feature["contours_shape"].bytes_list.value[0], out_type=tf.int64)
        == contours_shape
    )
