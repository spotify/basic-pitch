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

from basic_pitch.constants import AUDIO_N_SAMPLES, ANNOTATIONS_N_SEMITONES, ANNOT_N_FRAMES
from basic_pitch.visualize import visualize_transcription


def test_visualize_transcription(tmpdir: str) -> None:
    inputs = tf.random.normal([1, AUDIO_N_SAMPLES, 1])
    targets = {
        key: tf.random.normal([1, ANNOTATIONS_N_SEMITONES, ANNOT_N_FRAMES]) for key in ["onset", "contour", "note"]
    }
    outputs = {
        key: tf.random.normal([1, ANNOTATIONS_N_SEMITONES, ANNOT_N_FRAMES]) for key in ["onset", "contour", "note"]
    }

    visualize_transcription(
        file_writer=tf.summary.create_file_writer(str(tmpdir)),
        stage="train",
        inputs=inputs,
        targets=targets,
        outputs=outputs,
        loss=np.random.random(),
        step=1,
        sonify=True,
        contours=True,
    )
