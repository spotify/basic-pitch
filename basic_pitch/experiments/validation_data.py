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

import csv
from itertools import chain
import os

import mirdata
import sox


def load_validation_tracks(dataset):
    currdir = os.path.dirname(os.path.realpath(__file__))
    fname = "{}_validation_tracks.csv".format(dataset)
    with open(os.path.join(currdir, "split_ids", fname), "r") as fhandle:
        reader = csv.reader(fhandle)
        track_ids = [line[0] for line in reader]
    return track_ids


def maestro_tracks(data_home, limit=20):
    validation_tracks = load_validation_tracks("maestro")
    if limit:
        validation_tracks = validation_tracks[:limit]
    maestro = mirdata.initialize("maestro", data_home=data_home)
    for track_id in validation_tracks:
        track = maestro.track(track_id)
        yield ("maestro", track_id, "piano", track.audio_path, track.notes)


def guitarset_tracks(data_home):
    validation_tracks = load_validation_tracks("guitarset")
    guitarset = mirdata.initialize("guitarset", data_home=data_home)
    for track_id in validation_tracks:
        track = guitarset.track(track_id)
        yield (
            "guitarset",
            track_id,
            "guitar",
            track.audio_mic_path,
            track.notes_all,
        )


def slakh_tracks(data_home, limit=100):
    validation_tracks = load_validation_tracks("slakh")
    if limit:
        validation_tracks = validation_tracks[:limit]
    slakh = mirdata.initialize("slakh", data_home=data_home)
    for track_id in validation_tracks:
        track = slakh.track(track_id)
        if track.data_split != "test" or track.audio_path is None or track.is_drum or track.notes is None:
            continue

        yield ("slakh", track.track_id, track.instrument, track.audio_path, track.notes)


def medleydb_pitch_tracks(data_home):
    validation_tracks = load_validation_tracks("medleydb_pitch")
    mdb_pitch = mirdata.initialize("medleydb_pitch", data_home=data_home)
    for track_id in validation_tracks:
        track = mdb_pitch.track(track_id)
        if track.notes is None:
            continue
        yield (
            "medleydb_pitch",
            track_id,
            track.instrument,
            track.audio_path,
            track.notes_pyin,
        )


def ikala_tracks(data_home):
    validation_tracks = load_validation_tracks("ikala")
    ikala = mirdata.initialize("ikala", data_home=data_home)
    for track_id in validation_tracks:
        track = ikala.track(track_id)
        if track.notes is None:
            continue

        local_wav_path = track.audio_path + "_vocals.wav"
        if not os.path.exists(local_wav_path):
            tfm = sox.Transformer()
            tfm.rate(22050)
            tfm.remix({1: [2]})
            tfm.channels(1)
            tfm.build(track.audio_path, local_wav_path)

        yield ("ikala", track_id, "vocals", local_wav_path, track.notes_pyin)


def validation_data_generator(data_home):

    all_track_generator = chain(
        guitarset_tracks(None),
        slakh_tracks(None),
        medleydb_pitch_tracks(None),
        ikala_tracks(None),
        maestro_tracks(None),
    )
    return all_track_generator
