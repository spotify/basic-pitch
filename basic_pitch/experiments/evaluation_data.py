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


def load_test_tracks(dataset):
    currdir = os.path.dirname(os.path.realpath(__file__))
    fname = "{}_test_tracks.csv".format(dataset)
    with open(os.path.join(currdir, "split_ids", fname), "r") as fhandle:
        reader = csv.reader(fhandle)
        track_ids = [line[0] for line in reader]
    return track_ids


def update_data_home(data_home, dataset_name):
    if data_home is None:
        return data_home

    if data_home.startswith("gs://"):
        return data_home

    if dataset_name not in data_home:
        return os.path.join(data_home, dataset_name)


def maestro_tracks(data_home, limit=20):
    test_tracks = load_test_tracks("maestro")
    if limit:
        test_tracks = test_tracks[:limit]
    maestro = mirdata.initialize("maestro", data_home=data_home)
    for track_id in test_tracks:
        track = maestro.track(track_id)
        yield ("maestro", track_id, "piano", track.audio_path, track.notes)


def guitarset_tracks(data_home):
    test_tracks = load_test_tracks("guitarset")
    guitarset = mirdata.initialize("guitarset", data_home=data_home)
    for track_id in test_tracks:
        track = guitarset.track(track_id)
        yield (
            "guitarset",
            track_id,
            "guitar",
            track.audio_mic_path,
            track.notes_all,
        )


def slakh_tracks(data_home, limit=100):
    test_tracks = load_test_tracks("slakh")
    if limit:
        test_tracks = test_tracks[:limit]
    slakh = mirdata.initialize("slakh", data_home=data_home)
    for track_id in test_tracks:
        track = slakh.track(track_id)
        if track.data_split != "test" or track.audio_path is None or track.is_drum or track.notes is None:
            continue

        yield ("slakh", track.track_id, track.instrument, track.audio_path, track.notes)


# we don't use this for now
def slakh_tracks_mixes(data_home):
    test_tracks = load_test_tracks("slakh")
    mtrack_ids = list(set([t.split("-")[0] for t in test_tracks]))
    slakh = mirdata.initialize("slakh", data_home=data_home)
    for mtrack_id in mtrack_ids:
        mtrack = slakh.multitrack(mtrack_id)
        yield ("slakh", mtrack_id, "multi", mtrack.mix_path, mtrack.notes)


def phenicx_anechoic(data_home):
    phenix = mirdata.initialize("phenicx_anechoic", data_home=data_home)
    tracks = phenix.load_tracks()

    for track in tracks.values():
        target_submix_fpath = os.path.join(data_home, "mixes", "{}.wav".format(track.track_id))

        if len(track.audio_paths) > 1 and not os.path.exists(target_submix_fpath):
            fpath = target_submix_fpath
            cbn = sox.Combiner()
            cbn.build(track.audio_paths, fpath, "mix")
        else:
            fpath = track.audio_paths[0]

        yield ("phenicx", track.track_id, track.instrument, fpath, track.notes)


def phenicx_anechoic_mix(data_home):
    phenix = mirdata.initialize("phenicx_anechoic", data_home=data_home)
    mtracks = phenix.load_multitracks()
    for mtrack in mtracks.values():
        target_submix_fpath = os.path.join(data_home, "mixes", "{}.wav".format(mtrack.mtrack_id))

        if not os.path.exists(target_submix_fpath):
            audio_paths = []
            for track in mtrack.tracks.values():
                audio_paths.extend(track.audio_paths)
            cbn = sox.Combiner()
            cbn.build(audio_paths, target_submix_fpath, "mix")

        notes = mtrack.get_notes_target(list(mtrack.tracks.keys()))

        yield ("phenicx", mtrack.mtrack_id, "multi", target_submix_fpath, notes)


def dagstuhl_tracks_singlevoice(data_home):
    dagstuhl = mirdata.initialize("dagstuhl_choirset", data_home=data_home)
    for track in dagstuhl.load_tracks().values():
        if track.audio_hsm_path is None or track.score is None:
            continue

        yield (
            "dagstuhl_choirset",
            track.track_id,
            "vocals",
            track.audio_hsm_path,
            track.score,
        )


def dagsthul_tracks_choir(data_home):
    dagstuhl = mirdata.initialize("dagstuhl_choirset", data_home=data_home)
    mtracks = [mtrack for mtrack in dagstuhl.load_multitracks().values()]
    for mtrack in mtracks:
        no_score = False
        for track in mtrack.tracks.values():
            if track.score is None:
                no_score = True
                break

        if no_score:
            continue

        yield (
            "dagstuhl_choirset",
            mtrack.mtrack_id,
            "vocals-multi",
            mtrack.audio_rev_path,
            mtrack.notes,
        )


def evaluation_data_generator(data_home, maestro_limit=None, slakh_limit=None):
    all_track_generator = chain(
        guitarset_tracks(update_data_home(data_home, "guitarset")),
        slakh_tracks(update_data_home(data_home, "slakh"), limit=slakh_limit),
        dagstuhl_tracks_singlevoice(update_data_home(data_home, "dagstuhl_choirset")),
        dagsthul_tracks_choir(update_data_home(data_home, "dagstuhl_choirset")),
        phenicx_anechoic(update_data_home(data_home, "phenicx_anechoic")),
        phenicx_anechoic_mix(update_data_home(data_home, "phenicx_anechoic")),
        maestro_tracks(update_data_home(data_home, "maestro"), limit=maestro_limit),
    )
    return all_track_generator
