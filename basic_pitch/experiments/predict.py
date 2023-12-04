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

import argparse

import tensorflow as tf
import numpy as np
import librosa

from basic_pitch.constants import (
    AUDIO_SAMPLE_RATE,
    AUDIO_N_SAMPLES,
    ANNOTATIONS_FPS,
    FFT_HOP,
)
from basic_pitch import note_creation as infer


def window_audio_file(audio_original, hop_size):
    """
    Pad appropriately an audio file, and return as
    windowed signal, with window length = AUDIO_N_SAMPLES

    Returns
    -------
    audio_windowed: tensor with shape (n_windows, AUDIO_N_SAMPLES, 1)
        audio windowed into fixed length chunks
    window_times: list of {'start':.., 'end':...} objects (times in seconds)

    """

    audio_windowed = tf.expand_dims(
        tf.signal.frame(audio_original, AUDIO_N_SAMPLES, hop_size, pad_end=True, pad_value=0),
        axis=-1,
    )
    window_times = [
        {
            "start": t_start,
            "end": t_start + (AUDIO_N_SAMPLES / AUDIO_SAMPLE_RATE),
        }
        for t_start in np.arange(audio_windowed.shape[0]) * hop_size / AUDIO_SAMPLE_RATE
    ]
    return audio_windowed, window_times


def get_audio_input(audio_path, overlap_len, hop_size):
    """
    Read wave file (as mono), pad appropriately, and return as
    windowed signal, with window length = AUDIO_N_SAMPLES

    Returns
    -------
    audio_windowed: tensor with shape (n_windows, AUDIO_N_SAMPLES, 1)
        audio windowed into fixed length chunks
    window_times: list of {'start':.., 'end':...} objects (times in seconds)
    audio_original_length: int
        length of original audio file, in frames, BEFORE padding.

    """
    assert overlap_len % 2 == 0, "overlap_length must be even, got {}".format(overlap_len)
    audio_original, _ = librosa.load(audio_path, sr=AUDIO_SAMPLE_RATE, mono=True)
    original_length = audio_original.shape[0]
    audio_original = np.concatenate([np.zeros((int(overlap_len / 2),), dtype=np.float32), audio_original])
    audio_windowed, window_times = window_audio_file(audio_original, hop_size)
    return audio_windowed, window_times, original_length


def unwrap_output(output, audio_original_length, n_overlapping_frames):
    """Unwrap batched model predictions to a single matrix.

    Parameters:
        output : array (n_batches, n_times_short, n_freqs)
        audio_original_length : length of original audio signal (in samples)
        n_overlapping_frames : number of overlapping frames in the output

    Return:
        array (n_times, n_freqs)
    """
    raw_output = output.numpy()
    if len(raw_output.shape) != 3:
        return None

    n_olap = int(0.5 * n_overlapping_frames)
    if n_olap > 0:
        # remove half of the overlapping frames from beginning and end
        raw_output = raw_output[:, n_olap:-n_olap, :]

    output_shape = raw_output.shape
    n_output_frames_original = int(np.floor(audio_original_length * (ANNOTATIONS_FPS / AUDIO_SAMPLE_RATE)))
    unwrapped_output = raw_output.reshape(output_shape[0] * output_shape[1], output_shape[2])
    return unwrapped_output[:n_output_frames_original, :]  # trim to original audio length


def run_inference(audio_path, model):
    # overlap 30 frames  ## TODO compute exact receptive field
    n_overlapping_frames = 30
    overlap_len = n_overlapping_frames * FFT_HOP
    hop_size = AUDIO_N_SAMPLES - overlap_len

    audio_windowed, _, audio_original_length = get_audio_input(audio_path, overlap_len, hop_size)

    output = model(audio_windowed)
    unwrapped_output = {k: unwrap_output(output[k], audio_original_length, n_overlapping_frames) for k in output}
    return unwrapped_output


def main(audio_path, model_path, midi_save_path, sonify):
    model = tf.saved_model.load(model_path)

    output = run_inference(audio_path, model)

    if midi_save_path == "":
        print("No midi save path specified. To save to midi file, pass a value to --midi-save-path")
        midi_save_path = None

    mid = infer.model_output_to_midi(
        output,
        onset_thresh=0.5,
        frame_thresh=0.3,
        midi_path=midi_save_path,
    )
    if sonify and midi_save_path != "":
        infer.sonify_midi(mid, midi_save_path + ".wav")
    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process some integers.")
    parser.add_argument("audio_path", type=str, help="path to the input audio file")
    parser.add_argument("model_path", type=str, help="path to the saved model directory")
    parser.add_argument(
        "-mo",
        "--midi-save-path",
        type=str,
        default="",
        help="[MIDI only] Path to save midi file",
    )
    parser.add_argument(
        "--sonify",
        action="store_true",
        help="[MIDI only] sonify midi outputs.",
    )
    args = parser.parse_args()
    main(
        args.audio_path,
        args.model_path,
        args.midi_save_path,
        args.sonify,
    )
