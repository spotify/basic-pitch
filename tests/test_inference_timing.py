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

from typing import Optional

import numpy as np
import pytest

from basic_pitch import inference
from basic_pitch.constants import ANNOT_N_FRAMES, AUDIO_N_SAMPLES, AUDIO_SAMPLE_RATE, FFT_HOP


@pytest.mark.parametrize("n_overlapping_frames", [0, 30, 31])
@pytest.mark.parametrize("hop_adjustment", [0, -100])
def test_frame_samples_follow_audio_windows(
    monkeypatch: pytest.MonkeyPatch, n_overlapping_frames: int, hop_adjustment: int
) -> None:
    # Each sample encodes its original position, so sampling the actual input
    # windows gives an independent reference for the unwrapped frame positions.
    audio = np.arange(1, 3 * AUDIO_N_SAMPLES + 1, dtype=np.float32)
    monkeypatch.setattr(inference.librosa, "load", lambda *args, **kwargs: (audio, AUDIO_SAMPLE_RATE))
    overlap_len = n_overlapping_frames * FFT_HOP
    hop_size = AUDIO_N_SAMPLES - overlap_len + hop_adjustment
    predictions = []
    for window, _, original_length in inference.get_audio_input("unused.wav", overlap_len, hop_size):
        predictions.append(window[0, np.arange(ANNOT_N_FRAMES) * FFT_HOP, :])
    output = inference.unwrap_output(np.stack(predictions), original_length, n_overlapping_frames, hop_size)
    samples = inference.model_frames_to_samples(output.shape[0], n_overlapping_frames, hop_size)
    valid = output[:, 0] != 0  # Ignore frames that sample the zero padding.
    np.testing.assert_array_equal(samples[valid], output[valid, 0] - 1)
    assert samples.dtype == np.int64


def test_frame_samples_do_not_accumulate_drift() -> None:
    # A 530-second recording crosses more than 320 window boundaries.
    original_length = 530 * AUDIO_SAMPLE_RATE
    overlap_len = inference.DEFAULT_OVERLAPPING_FRAMES * FFT_HOP
    hop_size = AUDIO_N_SAMPLES - overlap_len
    window_starts = np.arange(0, original_length + overlap_len // 2, hop_size, dtype=np.int64)
    predictions = window_starts[:, None, None] + np.arange(ANNOT_N_FRAMES)[None, :, None] * FFT_HOP - overlap_len // 2
    output = inference.unwrap_output(predictions, original_length, inference.DEFAULT_OVERLAPPING_FRAMES, hop_size)
    samples = inference.model_frames_to_samples(output.shape[0])
    np.testing.assert_array_equal(samples, output[:, 0])
    naive_samples = np.arange(output.shape[0]) * FFT_HOP
    assert (naive_samples[-1] - samples[-1]) / AUDIO_SAMPLE_RATE > 2.7
    # The default seam is 188 samples shorter than a regular FFT_HOP.
    assert samples[142] == hop_size
    assert samples[142] - samples[141] == FFT_HOP - 188


def test_frame_samples_empty() -> None:
    samples = inference.model_frames_to_samples(0)
    assert samples.shape == (0,)
    assert samples.dtype == np.int64


@pytest.mark.parametrize(
    "n_frames,n_overlapping_frames,hop_size",
    [(-1, 30, None), (1, -1, None), (1, ANNOT_N_FRAMES, None), (1, 30, 0), (1, 30, -1)],
)
def test_frame_samples_invalid_parameters(n_frames: int, n_overlapping_frames: int, hop_size: Optional[int]) -> None:
    with pytest.raises(ValueError):
        inference.model_frames_to_samples(n_frames, n_overlapping_frames, hop_size)
