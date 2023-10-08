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

import numpy as np

FFT_HOP = 256
N_FFT = 8 * FFT_HOP

NOTES_BINS_PER_SEMITONE = 1
CONTOURS_BINS_PER_SEMITONE = 3
# base frequency of the CENTRAL bin of the first semitone (i.e., the
# second bin if annotations_bins_per_semitone is 3)
ANNOTATIONS_BASE_FREQUENCY = 27.5  # lowest key on a piano
ANNOTATIONS_N_SEMITONES = 88  # number of piano keys
AUDIO_SAMPLE_RATE = 22050
AUDIO_N_CHANNELS = 1
N_FREQ_BINS_NOTES = ANNOTATIONS_N_SEMITONES * NOTES_BINS_PER_SEMITONE
N_FREQ_BINS_CONTOURS = ANNOTATIONS_N_SEMITONES * CONTOURS_BINS_PER_SEMITONE

AUDIO_WINDOW_LENGTH = 2  # duration in seconds of training examples - original 1

ANNOTATIONS_FPS = AUDIO_SAMPLE_RATE // FFT_HOP
ANNOTATION_HOP = 1.0 / ANNOTATIONS_FPS

# ANNOT_N_TIME_FRAMES is the number of frames in the time-frequency representations we compute
ANNOT_N_FRAMES = ANNOTATIONS_FPS * AUDIO_WINDOW_LENGTH

# AUDIO_N_SAMPLES is the number of samples in the (clipped) audio that we use as input to the models
AUDIO_N_SAMPLES = AUDIO_SAMPLE_RATE * AUDIO_WINDOW_LENGTH - FFT_HOP

DATASET_SAMPLING_FREQUENCY = {
    "MAESTRO": 5,
    "GuitarSet": 2,
    "MedleyDB-Pitch": 2,
    "iKala": 2,
    "slakh": 2,
}


def _freq_bins(bins_per_semitone: int, base_frequency: float, n_semitones: int) -> np.array:
    d = 2.0 ** (1.0 / (12 * bins_per_semitone))
    bin_freqs = base_frequency * d ** np.arange(bins_per_semitone * n_semitones)
    return bin_freqs


FREQ_BINS_NOTES = _freq_bins(NOTES_BINS_PER_SEMITONE, ANNOTATIONS_BASE_FREQUENCY, ANNOTATIONS_N_SEMITONES)
FREQ_BINS_CONTOURS = _freq_bins(CONTOURS_BINS_PER_SEMITONE, ANNOTATIONS_BASE_FREQUENCY, ANNOTATIONS_N_SEMITONES)
