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

import pathlib
from collections import defaultdict
from typing import DefaultDict, Dict, List, Optional, Tuple, Union
import mir_eval
import librosa
import resampy
import numpy as np
import pretty_midi
import scipy
from scipy.io import wavfile

from basic_pitch.constants import (
    AUDIO_SAMPLE_RATE,
    ANNOTATIONS_N_SEMITONES,
    ANNOTATIONS_BASE_FREQUENCY,
    AUDIO_N_SAMPLES,
    ANNOT_N_FRAMES,
    CONTOURS_BINS_PER_SEMITONE,
    FFT_HOP,
    N_FREQ_BINS_CONTOURS,
)

MIDI_OFFSET = 21
SONIFY_FS = 3000
N_PITCH_BEND_TICKS = 8192
MAX_FREQ_IDX = 87


def model_output_to_notes(
    output: Dict[str, np.array],
    onset_thresh: float,
    frame_thresh: float,
    infer_onsets: bool = True,
    min_note_len: int = 11,
    min_freq: Optional[float] = None,
    max_freq: Optional[float] = None,
    include_pitch_bends: bool = True,
    multiple_pitch_bends: bool = False,
    melodia_trick: bool = True,
    midi_tempo: float = 120,
) -> Tuple[pretty_midi.PrettyMIDI, List[Tuple[float, float, int, float, Optional[List[int]]]]]:
    """Convert model output to MIDI

    Args:
        output: A dictionary with shape
            {
                'frame': array of shape (n_times, n_freqs),
                'onset': array of shape (n_times, n_freqs),
                'contour': array of shape (n_times, 3*n_freqs)
            }
            representing the output of the basic pitch model.
        onset_thresh: Minimum amplitude of an onset activation to be considered an onset.
        infer_onsets: If True, add additional onsets when there are large differences in frame amplitudes.
        min_note_len: The minimum allowed note length in frames.
        min_freq: Minimum allowed output frequency, in Hz. If None, all frequencies are used.
        max_freq: Maximum allowed output frequency, in Hz. If None, all frequencies are used.
        include_pitch_bends: If True, include pitch bends.
        multiple_pitch_bends: If True, allow overlapping notes in midi file to have pitch bends.
        melodia_trick: Use the melodia post-processing step.

    Returns:
        midi : pretty_midi.PrettyMIDI object
        note_events: A list of note event tuples (start_time_s, end_time_s, pitch_midi, amplitude)
    """
    frames = output["note"]
    onsets = output["onset"]
    contours = output["contour"]

    estimated_notes = output_to_notes_polyphonic(
        frames,
        onsets,
        onset_thresh=onset_thresh,
        frame_thresh=frame_thresh,
        infer_onsets=infer_onsets,
        min_note_len=min_note_len,
        min_freq=min_freq,
        max_freq=max_freq,
        melodia_trick=melodia_trick,
    )
    if include_pitch_bends:
        estimated_notes_with_pitch_bend = get_pitch_bends(contours, estimated_notes)
    else:
        estimated_notes_with_pitch_bend = [(note[0], note[1], note[2], note[3], None) for note in estimated_notes]

    times_s = model_frames_to_time(contours.shape[0])
    estimated_notes_time_seconds = [
        (times_s[note[0]], times_s[note[1]], note[2], note[3], note[4]) for note in estimated_notes_with_pitch_bend
    ]

    return (
        note_events_to_midi(estimated_notes_time_seconds, multiple_pitch_bends, midi_tempo),
        estimated_notes_time_seconds,
    )


def sonify_midi(midi: pretty_midi.PrettyMIDI, save_path: Union[pathlib.Path, str], sr: Optional[int] = 44100) -> None:
    """Sonify a pretty_midi midi object and save to a file.

    Args:
        midi: A pretty_midi.PrettyMIDI object that will be sonified.
        save_path: Where to save the sonified midi.
        sr: Sample rate for rendering audio from midi.
    """
    y = midi.synthesize(sr)
    wavfile.write(save_path, sr, y)


def sonify_salience(
    gram: np.array, semitone_resolution: float, save_path: Optional[str] = None, thresh: float = 0.2
) -> Tuple[np.array, int]:
    """Sonify a salience matrix.

    Args:
        gram: A matrix of pitch salience values with range 0-1, with shape (n_freqs, n_times).
            The frequencies are logarithmically spaced.
        semitone_resolution: The number of bins per semitone in gram.
        save_path: Optional location to save the sonified salience.
        thresh: Salience values below thresh will not be sonified. Used to increase the speed of this function.

    Returns:
        A tuple of the sonified salience as an audio signal and the associated sample rate.
    """
    freqs = librosa.core.cqt_frequencies(
        n_bins=ANNOTATIONS_N_SEMITONES * semitone_resolution,
        fmin=ANNOTATIONS_BASE_FREQUENCY,
        bins_per_octave=12 * semitone_resolution,
    )
    # this function is slow - for speed, only sonify frequencies below
    # sonify_fs/2 Hz (e.g. 1000 Hz)
    max_freq_idx = np.where(freqs > SONIFY_FS / 2)[0][0]
    times = librosa.core.frames_to_time(
        np.arange(gram.shape[1]),
        sr=AUDIO_SAMPLE_RATE,
        hop_length=AUDIO_N_SAMPLES / ANNOT_N_FRAMES,  # THIS IS THE CORRECT HOP!!
    )
    gram[gram < thresh] = 0
    y = mir_eval.sonify.time_frequency(gram[:max_freq_idx, :], freqs[:max_freq_idx], times, fs=SONIFY_FS)
    if save_path:
        y_resamp = resampy.resample(y, SONIFY_FS, 44100)
        wavfile.write(save_path, 44100, y_resamp)

    return y, SONIFY_FS


def midi_pitch_to_contour_bin(pitch_midi: int) -> np.array:
    """Convert midi pitch to conrresponding index in contour matrix

    Args:
        pitch_midi: pitch in midi

    Returns:
        index in contour matrix

    """
    pitch_hz = librosa.midi_to_hz(pitch_midi)
    return 12.0 * CONTOURS_BINS_PER_SEMITONE * np.log2(pitch_hz / ANNOTATIONS_BASE_FREQUENCY)


def get_pitch_bends(
    contours: np.ndarray, note_events: List[Tuple[int, int, int, float]], n_bins_tolerance: int = 25
) -> List[Tuple[int, int, int, float, Optional[List[int]]]]:
    """Given note events and contours, estimate pitch bends per note.
    Pitch bends are represented as a sequence of evenly spaced midi pitch bend control units.
    The time stamps of each pitch bend can be inferred by computing an evenly spaced grid between
    the start and end times of each note event.

    Args:
        contours: Matrix of estimated pitch contours
        note_events: note event tuple
        n_bins_tolerance: Pitch bend estimation range. Defaults to 25.

    Returns:
        note events with pitch bends
    """
    window_length = n_bins_tolerance * 2 + 1
    freq_gaussian = scipy.signal.gaussian(window_length, std=5)
    note_events_with_pitch_bends = []
    for start_idx, end_idx, pitch_midi, amplitude in note_events:
        freq_idx = int(np.round(midi_pitch_to_contour_bin(pitch_midi)))
        freq_start_idx = np.max([freq_idx - n_bins_tolerance, 0])
        freq_end_idx = np.min([N_FREQ_BINS_CONTOURS, freq_idx + n_bins_tolerance + 1])

        pitch_bend_submatrix = (
            contours[start_idx:end_idx, freq_start_idx:freq_end_idx]
            * freq_gaussian[
                np.max([0, n_bins_tolerance - freq_idx]) : window_length
                - np.max([0, freq_idx - (N_FREQ_BINS_CONTOURS - n_bins_tolerance - 1)])
            ]
        )
        pb_shift = n_bins_tolerance - np.max([0, n_bins_tolerance - freq_idx])

        bends: Optional[List[int]] = list(
            np.argmax(pitch_bend_submatrix, axis=1) - pb_shift
        )  # this is in units of 1/3 semitones
        note_events_with_pitch_bends.append((start_idx, end_idx, pitch_midi, amplitude, bends))
    return note_events_with_pitch_bends


def note_events_to_midi(
    note_events_with_pitch_bends: List[Tuple[float, float, int, float, Optional[List[int]]]],
    multiple_pitch_bends: bool = False,
    midi_tempo: float = 120,
) -> pretty_midi.PrettyMIDI:
    """Create a pretty_midi object from note events

    Args:
        note_events : list of tuples [(start_time_seconds, end_time_seconds, pitch_midi, amplitude)]
            where amplitude is a number between 0 and 1
        multiple_pitch_bends : If True, allow overlapping notes to have pitch bends
            Note: this will assign each pitch to its own midi instrument, as midi does not yet
            support per-note pitch bends

    Returns:
        pretty_midi.PrettyMIDI() object

    """
    mid = pretty_midi.PrettyMIDI(initial_tempo=midi_tempo)
    if not multiple_pitch_bends:
        note_events_with_pitch_bends = drop_overlapping_pitch_bends(note_events_with_pitch_bends)

    piano_program = pretty_midi.instrument_name_to_program("Electric Piano 1")
    instruments: DefaultDict[int, pretty_midi.Instrument] = defaultdict(
        lambda: pretty_midi.Instrument(program=piano_program)
    )
    for start_time, end_time, note_number, amplitude, pitch_bend in note_events_with_pitch_bends:
        instrument = instruments[note_number] if multiple_pitch_bends else instruments[0]
        note = pretty_midi.Note(
            velocity=int(np.round(127 * amplitude)),
            pitch=note_number,
            start=start_time,
            end=end_time,
        )
        instrument.notes.append(note)
        if not pitch_bend:
            continue
        pitch_bend_times = np.linspace(start_time, end_time, len(pitch_bend))
        pitch_bend_midi_ticks = np.round(np.array(pitch_bend) * 4096 / CONTOURS_BINS_PER_SEMITONE).astype(int)
        # This supports pitch bends up to 2 semitones
        # If we estimate pitch bends above/below 2 semitones, crop them here when adding them to the midi file
        pitch_bend_midi_ticks[pitch_bend_midi_ticks > N_PITCH_BEND_TICKS - 1] = N_PITCH_BEND_TICKS - 1
        pitch_bend_midi_ticks[pitch_bend_midi_ticks < -N_PITCH_BEND_TICKS] = -N_PITCH_BEND_TICKS
        for pb_time, pb_midi in zip(pitch_bend_times, pitch_bend_midi_ticks):
            instrument.pitch_bends.append(pretty_midi.PitchBend(pb_midi, pb_time))
    mid.instruments.extend(instruments.values())

    return mid


def drop_overlapping_pitch_bends(
    note_events_with_pitch_bends: List[Tuple[float, float, int, float, Optional[List[int]]]]
) -> List[Tuple[float, float, int, float, Optional[List[int]]]]:
    """Drop pitch bends from any notes that overlap in time with another note"""
    note_events = sorted(note_events_with_pitch_bends)
    for i in range(len(note_events) - 1):
        for j in range(i + 1, len(note_events)):
            if note_events[j][0] >= note_events[i][1]:  # start j > end i
                break
            note_events[i] = note_events[i][:-1] + (None,)  # last field is pitch bend
            note_events[j] = note_events[j][:-1] + (None,)

    return note_events


def get_infered_onsets(onsets: np.array, frames: np.array, n_diff: int = 2) -> np.array:
    """Infer onsets from large changes in frame amplitudes.

    Args:
        onsets: Array of note onset predictions.
        frames: Audio frames.
        n_diff: Differences used to detect onsets.

    Returns:
        The maximum between the predicted onsets and its differences.
    """
    diffs = []
    for n in range(1, n_diff + 1):
        frames_appended = np.concatenate([np.zeros((n, frames.shape[1])), frames])
        diffs.append(frames_appended[n:, :] - frames_appended[:-n, :])
    frame_diff = np.min(diffs, axis=0)
    frame_diff[frame_diff < 0] = 0
    frame_diff[:n_diff, :] = 0
    frame_diff = np.max(onsets) * frame_diff / np.max(frame_diff)  # rescale to have the same max as onsets

    max_onsets_diff = np.max([onsets, frame_diff], axis=0)  # use the max of the predicted onsets and the differences

    return max_onsets_diff


def constrain_frequency(
    onsets: np.array, frames: np.array, max_freq: Optional[float], min_freq: Optional[float]
) -> Tuple[np.array, np.array]:
    """Zero out activations above or below the max/min frequencies

    Args:
        onsets: Onset activation matrix (n_times, n_freqs)
        frames: Frame activation matrix (n_times, n_freqs)
        max_freq: The maximum frequency to keep.
        min_freq: the minimum frequency to keep.

    Returns:
       The onset and frame activation matrices, with frequencies outside the min and max
       frequency set to 0.
    """
    if max_freq is not None:
        max_freq_idx = int(np.round(librosa.hz_to_midi(max_freq) - MIDI_OFFSET))
        onsets[:, max_freq_idx:] = 0
        frames[:, max_freq_idx:] = 0
    if min_freq is not None:
        min_freq_idx = int(np.round(librosa.hz_to_midi(min_freq) - MIDI_OFFSET))
        onsets[:, :min_freq_idx] = 0
        frames[:, :min_freq_idx] = 0

    return onsets, frames


def model_frames_to_time(n_frames: int) -> np.ndarray:
    original_times = librosa.core.frames_to_time(
        np.arange(n_frames),
        sr=AUDIO_SAMPLE_RATE,
        hop_length=FFT_HOP,
    )
    window_numbers = np.floor(np.arange(n_frames) / ANNOT_N_FRAMES)
    window_offset = (FFT_HOP / AUDIO_SAMPLE_RATE) * (
        ANNOT_N_FRAMES - (AUDIO_N_SAMPLES / FFT_HOP)
    ) + 0.0018  # this is a magic number, but it's needed for this to align properly
    times = original_times - (window_offset * window_numbers)
    return times


def output_to_notes_polyphonic(
    frames: np.array,
    onsets: np.array,
    onset_thresh: float,
    frame_thresh: float,
    min_note_len: int,
    infer_onsets: bool,
    max_freq: Optional[float],
    min_freq: Optional[float],
    melodia_trick: bool = True,
    energy_tol: int = 11,
) -> List[Tuple[int, int, int, float]]:
    """Decode raw model output to polyphonic note events

    Args:
        frames: Frame activation matrix (n_times, n_freqs).
        onsets: Onset activation matrix (n_times, n_freqs).
        onset_thresh: Minimum amplitude of an onset activation to be considered an onset.
        frame_thresh: Minimum amplitude of a frame activation for a note to remain "on".
        min_note_len: Minimum allowed note length in frames.
        infer_onsets: If True, add additional onsets when there are large differences in frame amplitudes.
        max_freq: Maximum allowed output frequency, in Hz.
        min_freq: Minimum allowed output frequency, in Hz.
        melodia_trick : Whether to use the melodia trick to better detect notes.
        energy_tol: Drop notes below this energy.

    Returns:
        list of tuples [(start_time_frames, end_time_frames, pitch_midi, amplitude)]
        representing the note events, where amplitude is a number between 0 and 1
    """

    n_frames = frames.shape[0]

    onsets, frames = constrain_frequency(onsets, frames, max_freq, min_freq)
    # use onsets inferred from frames in addition to the predicted onsets
    if infer_onsets:
        onsets = get_infered_onsets(onsets, frames)

    peak_thresh_mat = np.zeros(onsets.shape)
    peaks = scipy.signal.argrelmax(onsets, axis=0)
    peak_thresh_mat[peaks] = onsets[peaks]

    onset_idx = np.where(peak_thresh_mat >= onset_thresh)
    onset_time_idx = onset_idx[0][::-1]  # sort to go backwards in time
    onset_freq_idx = onset_idx[1][::-1]  # sort to go backwards in time

    remaining_energy = np.zeros(frames.shape)
    remaining_energy[:, :] = frames[:, :]

    # loop over onsets
    note_events = []
    for note_start_idx, freq_idx in zip(onset_time_idx, onset_freq_idx):
        # if we're too close to the end of the audio, continue
        if note_start_idx >= n_frames - 1:
            continue

        # find time index at this frequency band where the frames drop below an energy threshold
        i = note_start_idx + 1
        k = 0  # number of frames since energy dropped below threshold
        while i < n_frames - 1 and k < energy_tol:
            if remaining_energy[i, freq_idx] < frame_thresh:
                k += 1
            else:
                k = 0
            i += 1

        i -= k  # go back to frame above threshold

        # if the note is too short, skip it
        if i - note_start_idx <= min_note_len:
            continue

        remaining_energy[note_start_idx:i, freq_idx] = 0
        if freq_idx < MAX_FREQ_IDX:
            remaining_energy[note_start_idx:i, freq_idx + 1] = 0
        if freq_idx > 0:
            remaining_energy[note_start_idx:i, freq_idx - 1] = 0

        # add the note
        amplitude = np.mean(frames[note_start_idx:i, freq_idx])
        note_events.append(
            (
                note_start_idx,
                i,
                freq_idx + MIDI_OFFSET,
                amplitude,
            )
        )

    if melodia_trick:
        energy_shape = remaining_energy.shape

        while np.max(remaining_energy) > frame_thresh:
            i_mid, freq_idx = np.unravel_index(np.argmax(remaining_energy), energy_shape)
            remaining_energy[i_mid, freq_idx] = 0

            # forward pass
            i = i_mid + 1
            k = 0
            while i < n_frames - 1 and k < energy_tol:
                if remaining_energy[i, freq_idx] < frame_thresh:
                    k += 1
                else:
                    k = 0

                remaining_energy[i, freq_idx] = 0
                if freq_idx < MAX_FREQ_IDX:
                    remaining_energy[i, freq_idx + 1] = 0
                if freq_idx > 0:
                    remaining_energy[i, freq_idx - 1] = 0

                i += 1

            i_end = i - 1 - k  # go back to frame above threshold

            # backward pass
            i = i_mid - 1
            k = 0
            while i > 0 and k < energy_tol:
                if remaining_energy[i, freq_idx] < frame_thresh:
                    k += 1
                else:
                    k = 0

                remaining_energy[i, freq_idx] = 0
                if freq_idx < MAX_FREQ_IDX:
                    remaining_energy[i, freq_idx + 1] = 0
                if freq_idx > 0:
                    remaining_energy[i, freq_idx - 1] = 0

                i -= 1

            i_start = i + 1 + k  # go back to frame above threshold
            assert i_start >= 0, "{}".format(i_start)
            assert i_end < n_frames

            if i_end - i_start <= min_note_len:
                # note is too short, skip it
                continue

            # add the note
            amplitude = np.mean(frames[i_start:i_end, freq_idx])
            note_events.append(
                (
                    i_start,
                    i_end,
                    freq_idx + MIDI_OFFSET,
                    amplitude,
                )
            )

    return note_events
