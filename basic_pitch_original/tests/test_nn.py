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

from typing import Iterator, Tuple
import unittest

import numpy as np
import tensorflow as tf

from basic_pitch import models, nn
from basic_pitch.constants import (
    ANNOT_N_FRAMES,
    ANNOTATIONS_N_SEMITONES,
    AUDIO_N_SAMPLES,
    N_FREQ_BINS_CONTOURS,
)

BATCH_SIZE = 3

tfkl = tf.keras.layers


class TestHarmonicStacking(unittest.TestCase):
    def _audio_data_gen(self) -> Iterator[Tuple[np.array, np.array]]:
        while True:
            audio = np.random.uniform(size=(BATCH_SIZE, AUDIO_N_SAMPLES, 1)).astype(np.float32)
            output = np.random.uniform(size=(BATCH_SIZE, ANNOT_N_FRAMES, ANNOTATIONS_N_SEMITONES * 3, 1)).astype(
                np.float32
            )
            yield (audio, output)

    def _dummy_dataset(self) -> tf.data.Dataset:
        ds = tf.data.Dataset.from_generator(
            self._audio_data_gen,
            (tf.float32, tf.float32),
            (
                tf.TensorShape([BATCH_SIZE, AUDIO_N_SAMPLES, 1]),
                tf.TensorShape([BATCH_SIZE, ANNOT_N_FRAMES, ANNOTATIONS_N_SEMITONES * 3, 1]),
            ),
        )
        return ds

    def test_defaults(self) -> None:
        inputs = tf.keras.Input(shape=(AUDIO_N_SAMPLES, 1))
        x = models.get_cqt(inputs, 5, True)
        conv = nn.HarmonicStacking(3, [1, 2, 3, 4, 5], N_FREQ_BINS_CONTOURS)(x)
        outputs = tfkl.Conv2D(1, (3, 3), padding="same")(conv)
        model = tf.keras.Model(inputs=inputs, outputs=outputs)

        train_ds = self._dummy_dataset()
        valid_ds = self._dummy_dataset()

        model.compile(
            loss="binary_crossentropy",
            optimizer=tf.keras.optimizers.Adam(0.1),
            sample_weight_mode=None,
        )

        model.fit(
            train_ds,
            epochs=2,
            steps_per_epoch=2,
            validation_data=valid_ds,
            validation_steps=1,
        )

        for element in train_ds:
            in_vals = element[0]
            out_vals = model(in_vals)
            assert out_vals.shape == (
                BATCH_SIZE,
                ANNOT_N_FRAMES,
                ANNOTATIONS_N_SEMITONES * 3,
                1,
            )
            break

    def test_fractions(self) -> None:
        inputs = tf.keras.Input(shape=(AUDIO_N_SAMPLES, 1))
        x = models.get_cqt(inputs, 2, True)
        conv = nn.HarmonicStacking(3, [0.5, 1, 2], N_FREQ_BINS_CONTOURS)(x)
        outputs = tfkl.Conv2D(1, (3, 3), padding="same")(conv)
        model = tf.keras.Model(inputs=inputs, outputs=outputs)

        train_ds = self._dummy_dataset()
        valid_ds = self._dummy_dataset()

        model.compile(
            loss="binary_crossentropy",
            optimizer=tf.keras.optimizers.Adam(0.1),
            sample_weight_mode=None,
        )

        model.fit(
            train_ds,
            epochs=2,
            steps_per_epoch=2,
            validation_data=valid_ds,
            validation_steps=1,
        )

        for element in train_ds:
            inputs = element[0]
            output = model(inputs)
            assert output.shape == (
                BATCH_SIZE,
                ANNOT_N_FRAMES,
                ANNOTATIONS_N_SEMITONES * 3,
                1,
            )
            break
