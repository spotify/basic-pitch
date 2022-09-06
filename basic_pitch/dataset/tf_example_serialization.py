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

from typing import List, Tuple, Union

import sox
import numpy as np
import tensorflow as tf
from basic_pitch.constants import AUDIO_N_CHANNELS, AUDIO_SAMPLE_RATE


def int64_feature(value: Union[List[int], int]) -> tf.train.Feature:
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def float_feature(value: Union[List[float], float]) -> tf.train.Feature:
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def bytes_feature(value: Union[tf.Tensor, List[bytes], bytes]) -> tf.train.Feature:
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy()
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))


def _to_transcription_tfex(
    file_id: str,
    source: str,
    encoded_wav: bytes,
    notes_indices: List[Tuple[int, int]],
    notes_values: List[float],
    onsets_indices: List[float],
    onsets_values: List[float],
    contours_indices: List[float],
    contours_values: List[float],
    notes_onsets_shape: Tuple[int, int],
    contours_shape: Tuple[int, int],
) -> tf.train.Example:
    return tf.train.Example(
        features=tf.train.Features(
            feature={
                "file_id": bytes_feature(bytes(file_id, "utf-8")),
                "source": bytes_feature(bytes(source, "utf-8")),
                "audio_wav": bytes_feature(encoded_wav),
                "notes_indices": bytes_feature(tf.io.serialize_tensor(np.array(notes_indices, np.int64))),
                "notes_values": bytes_feature(tf.io.serialize_tensor(np.array(notes_values, np.float32))),
                "onsets_indices": bytes_feature(tf.io.serialize_tensor(np.array(onsets_indices, np.int64))),
                "onsets_values": bytes_feature(tf.io.serialize_tensor(np.array(onsets_values, np.float32))),
                "contours_indices": bytes_feature(tf.io.serialize_tensor(np.array(contours_indices, np.int64))),
                "contours_values": bytes_feature(tf.io.serialize_tensor(np.array(contours_values, np.float32))),
                "notes_onsets_shape": bytes_feature(tf.io.serialize_tensor(np.array(notes_onsets_shape, np.int64))),
                "contours_shape": bytes_feature(tf.io.serialize_tensor(np.array(contours_shape, np.int64))),
            }
        )
    )


def to_transcription_tfexample(
    file_id: str,
    source: str,
    audio_wav_file_path: str,
    notes_indices: List[Tuple[int, int]],
    notes_values: List[float],
    onsets_indices: List[float],
    onsets_values: List[float],
    contours_indices: List[float],
    contours_values: List[float],
    notes_onsets_shape: Tuple[int, int],
    contours_shape: Tuple[int, int],
):
    """
    - `file_id` string
    - `source` string  (e.g., "maestro")
    - `audio_file_path` path to a local WAV file (must be 22kHz stereo, checked)
    - `notes_indices` [(time, freq)], integers
    - `notes_values` [float]
    - `onsets_indices` same as above
    - `onsets_values` same as above
    - `contours_indices` same as above
    - `contours_values` same as above
    - `notes_onsets_shape` (time, freq), ints
    - `contours_onsets_shape` (time, freq), ints
    """
    assert sox.file_info.sample_rate(audio_wav_file_path) == AUDIO_SAMPLE_RATE
    assert sox.file_info.channels(audio_wav_file_path) == AUDIO_N_CHANNELS
    encoded_wav = open(audio_wav_file_path, "rb").read()
    return _to_transcription_tfex(
        file_id,
        source,
        encoded_wav,
        notes_indices,
        notes_values,
        onsets_indices,
        onsets_values,
        contours_indices,
        contours_values,
        notes_onsets_shape,
        contours_shape,
    )
