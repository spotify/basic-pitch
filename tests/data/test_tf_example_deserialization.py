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
import pathlib
import tensorflow as tf


from basic_pitch.data.tf_example_deserialization import sample_datasets, transcription_file_generator


def test_prepare_dataset():
    pass


def test_sample_datasets():
    pass


def test_transcription_file_generator(tmpdir: str):
    print("FUCK YOU ")
    file_gen, random_seed = transcription_file_generator("train", ["test2"], datasets_base_path=tmpdir, sample_weights=np.ndarray(1))
    assert random_seed is False

    print(file_gen())
