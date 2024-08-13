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

from typing import Dict

from basic_pitch.callbacks import VisualizeCallback
from basic_pitch.constants import AUDIO_N_SAMPLES, ANNOTATIONS_N_SEMITONES, ANNOT_N_FRAMES


class MockModel(tf.keras.Model):
    def __init__(self) -> None:
        super(MockModel, self).__init__()

    def call(self, inputs: tf.Tensor) -> Dict[str, tf.Tensor]:
        return {
            key: tf.random.normal((1, ANNOTATIONS_N_SEMITONES, ANNOT_N_FRAMES)) for key in ["onset", "contour", "note"]
        }


def create_mock_dataset() -> tf.data.Dataset:
    batch_size = 1
    inputs = tf.random.normal((batch_size, AUDIO_N_SAMPLES, 1))
    targets = {
        key: tf.random.normal((batch_size, ANNOTATIONS_N_SEMITONES, ANNOT_N_FRAMES))
        for key in ["onset", "contour", "note"]
    }
    dataset = tf.data.Dataset.from_tensor_slices((inputs, targets))
    dataset = dataset.batch(batch_size)
    return dataset


def test_visualize_callback_on_epoch_end(tmpdir: str) -> None:
    model = MockModel()
    model.compile(optimizer="adam", loss="mse")

    batch_size = 2  # needs to be at least 2 bc validation_split required

    x_train = np.random.random((batch_size, AUDIO_N_SAMPLES, 1))
    y_train = {
        key: np.random.random((batch_size, ANNOTATIONS_N_SEMITONES, ANNOT_N_FRAMES))
        for key in ["onset", "contour", "note"]
    }

    vc = VisualizeCallback(
        train_ds=create_mock_dataset(),
        validation_ds=create_mock_dataset(),
        tensorboard_dir=str(tmpdir),
        original_validation_ds=create_mock_dataset(),
        contours=True,
    )

    history = model.fit(x_train, y_train, epochs=1, validation_split=0.5, callbacks=[vc])
    assert history
