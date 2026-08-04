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

from typing import Optional

import numpy as np
import numpy.typing as npt

from basic_pitch.constants import ANNOT_N_FRAMES, AUDIO_N_SAMPLES, FFT_HOP


DEFAULT_OVERLAPPING_FRAMES = 30


def model_output_frame_samples(
    n_frames: int,
    n_overlapping_frames: int = DEFAULT_OVERLAPPING_FRAMES,
    hop_size: Optional[int] = None,
) -> npt.NDArray[np.int64]:
    """Return original-audio sample positions for unwrapped model frames.

    ``unwrap_output`` removes half of the overlapping frames from each side
    of every model window, then concatenates the retained blocks. The input
    window hop is not necessarily equal to the duration represented by one
    retained block, so globally treating frame ``i`` as ``i * FFT_HOP`` can
    accumulate timing drift across window boundaries.

    The initial half-overlap of zero padding cancels the removed leading
    model frames. A retained frame ``j`` from input window ``w`` therefore
    corresponds to original-audio sample ``w * hop_size + j * FFT_HOP``.

    Args:
        n_frames: Number of unwrapped output frames to map.
        n_overlapping_frames: Total number of output frames removed from each
            window, split evenly between its beginning and end.
        hop_size: Input-window hop in samples. If omitted, use the same value
            as ``run_inference`` for the requested overlap.

    Returns:
        A one-dimensional array of original-audio sample positions.

    Raises:
        ValueError: If an argument cannot describe a valid unwrapped output.
    """
    if n_frames < 0:
        raise ValueError(f"n_frames must be non-negative, got {n_frames}")
    if n_overlapping_frames < 0 or n_overlapping_frames >= ANNOT_N_FRAMES:
        raise ValueError(
            "n_overlapping_frames must be between 0 and "
            f"{ANNOT_N_FRAMES - 1}, got {n_overlapping_frames}"
        )
    if n_overlapping_frames % 2:
        raise ValueError(
            "n_overlapping_frames must be even so it can be removed equally "
            "from both sides of each window"
        )

    frames_per_window = ANNOT_N_FRAMES - n_overlapping_frames
    if hop_size is None:
        hop_size = AUDIO_N_SAMPLES - n_overlapping_frames * FFT_HOP
    if hop_size <= 0:
        raise ValueError(f"hop_size must be positive, got {hop_size}")

    frame_indices = np.arange(n_frames, dtype=np.int64)
    window_indices, within_window_indices = np.divmod(
        frame_indices, frames_per_window
    )
    return window_indices * hop_size + within_window_indices * FFT_HOP
