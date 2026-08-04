#!/usr/bin/env python
# encoding: utf-8
#
# Copyright 2026 Spotify AB
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
import pytest

from basic_pitch.constants import ANNOT_N_FRAMES, AUDIO_N_SAMPLES, FFT_HOP
from basic_pitch.frame_alignment import model_output_frame_samples


def test_model_output_frame_samples_accounts_for_window_hop() -> None:
    n_overlapping_frames = 30
    hop_size = AUDIO_N_SAMPLES - n_overlapping_frames * FFT_HOP
    frames_per_window = ANNOT_N_FRAMES - n_overlapping_frames

    samples = model_output_frame_samples(frames_per_window + 2)

    np.testing.assert_array_equal(
        samples[[0, frames_per_window - 1, frames_per_window, frames_per_window + 1]],
        [
            0,
            (frames_per_window - 1) * FFT_HOP,
            hop_size,
            hop_size + FFT_HOP,
        ],
    )
    assert samples[frames_per_window] - samples[frames_per_window - 1] != FFT_HOP


def test_model_output_frame_samples_supports_custom_alignment() -> None:
    np.testing.assert_array_equal(
        model_output_frame_samples(5, n_overlapping_frames=0, hop_size=1000),
        [0, FFT_HOP, 2 * FFT_HOP, 3 * FFT_HOP, 4 * FFT_HOP],
    )


@pytest.mark.parametrize(
    ("n_frames", "n_overlapping_frames", "hop_size"),
    [
        (-1, 30, None),
        (1, -2, None),
        (1, 29, None),
        (1, ANNOT_N_FRAMES, None),
        (1, 30, 0),
    ],
)
def test_model_output_frame_samples_rejects_invalid_alignment(
    n_frames: int, n_overlapping_frames: int, hop_size: int
) -> None:
    with pytest.raises(ValueError):
        model_output_frame_samples(n_frames, n_overlapping_frames, hop_size)
