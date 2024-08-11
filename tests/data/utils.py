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

import logging
import numpy as np
import pathlib
import soundfile as sf
import wave

from mido import MidiFile, MidiTrack, Message


def create_mock_wav(output_fpath: pathlib.Path, duration_min: int) -> None:
    assert output_fpath.suffix == ".wav"

    duration_seconds = duration_min * 60
    sample_rate = 44100
    n_channels = 2  # Stereo
    sampwidth = 2  # 2 bytes per sample (16-bit audio)

    # Generate a silent audio data array
    num_samples = duration_seconds * sample_rate
    audio_data = np.zeros((num_samples, n_channels), dtype=np.int16)

    # Create the WAV file
    with wave.open(str(output_fpath), "w") as wav_file:
        wav_file.setnchannels(n_channels)
        wav_file.setsampwidth(sampwidth)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(audio_data.tobytes())

    logging.info(f"Mock {duration_min}-minute WAV file '{output_fpath}' created successfully.")


def create_mock_flac(output_fpath: pathlib.Path) -> None:
    assert output_fpath.suffix == ".flac"

    frequency = 440  # A4
    duration = 2  # seconds
    sample_rate = 44100  # standard
    amplitude = 0.5

    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    sin_wave = amplitude * np.sin(duration * np.pi * frequency * t)

    # Save as a FLAC file
    sf.write(str(output_fpath), sin_wave, frequency, format="FLAC")

    logging.info(f"Mock FLAC file {str(output_fpath)} created successfully.")


def create_mock_midi(output_fpath: pathlib.Path) -> None:
    assert output_fpath.suffix in (".mid", ".midi")
    # Create a new MIDI file with one track
    mid = MidiFile()
    track = MidiTrack()
    mid.tracks.append(track)

    # Define a sequence of notes (time, type, note, velocity)
    notes = [
        (0, "note_on", 60, 64),  # C4
        (500, "note_off", 60, 64),
        (0, "note_on", 62, 64),  # D4
        (500, "note_off", 62, 64),
    ]

    # Add the notes to the track
    for time, type, note, velocity in notes:
        track.append(Message(type, note=note, velocity=velocity, time=time))

    # Save the MIDI file
    mid.save(output_fpath)

    logging.info(f"Mock MIDI file '{output_fpath}' created successfully.")
