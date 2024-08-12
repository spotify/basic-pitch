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
import tensorflow as tf

from basic_pitch.constants import AUDIO_N_SAMPLES
from basic_pitch.visualize import MAX_OUTPUTS, visualize_transcription


def test_visualize_transcription(tmpdir: str) -> None:
    # Mock Input Audio Tensor
    inputs = tf.random.uniform([MAX_OUTPUTS, AUDIO_N_SAMPLES, 1], minval=-1.0, maxval=1.0)

    # Mock Target and Output Tensors
    targets = {
        "onset": tf.random.uniform([MAX_OUTPUTS, 100, 128], minval=0.0, maxval=1.0),
        "contour": tf.random.uniform([MAX_OUTPUTS, 100, 128], minval=0.0, maxval=1.0),
        "note": tf.random.uniform([MAX_OUTPUTS, 100, 128], minval=0.0, maxval=1.0),
    }
    outputs = {
        "onset": tf.random.uniform([MAX_OUTPUTS, 100, 128], minval=0.0, maxval=1.0),
        "contour": tf.random.uniform([MAX_OUTPUTS, 100, 128], minval=0.0, maxval=1.0),
        "note": tf.random.uniform([MAX_OUTPUTS, 100, 128], minval=0.0, maxval=1.0),
    }

    # Mock loss value
    loss = np.random.random()

    # Mock step (epoch)
    step = 1

    # File writer (TensorBoard)
    file_writer = tf.summary.create_file_writer(str(tmpdir))

    visualize_transcription(
        file_writer=file_writer,
        stage="train",
        inputs=inputs,
        targets=targets,
        outputs=outputs,
        loss=loss,
        step=step,
        sonify=True,
        contours=True,
    )
