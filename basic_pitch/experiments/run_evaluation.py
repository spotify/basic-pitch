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
import json
import logging
import os

import librosa
import mir_eval
import numpy as np
import tensorflow as tf
from mirdata import io

from basic_pitch import note_creation

from predict import run_inference
from evaluation_data import evaluation_data_generator

logger = logging.getLogger("mirdata")
logger.setLevel(logging.ERROR)


def model_inference(audio_path, model, save_path):

    output = run_inference(audio_path, model)
    frames = output["note"]
    onsets = output["onset"]

    estimated_notes = note_creation.output_to_notes_polyphonic(
        frames,
        onsets,
        onset_thresh=0.5,
        frame_thresh=0.3,
        infer_onsets=True,
    )
    # [(start_time_seconds, end_time_seconds, pitch_midi, amplitude)]
    intervals = np.array([[n[0], n[1]] for n in estimated_notes])
    pitch_hz = librosa.midi_to_hz(np.array([n[2] for n in estimated_notes]))

    note_creation.note_events_to_midi(estimated_notes, save_path)

    return intervals, pitch_hz


def main(model_name: str, data_home: str) -> None:
    model_path = "tmp_models/{}".format(model_name)
    model = tf.saved_model.load(model_path)

    save_dir = os.path.join("model_outputs", model_name)

    all_track_generator = evaluation_data_generator(data_home)
    scores = {}
    for dataset, track_id, instrument, audio_path, note_data in all_track_generator:
        print("[{}] {}: {}".format(dataset, track_id, instrument))
        save_path = os.path.join(save_dir, "{}.mid".format(track_id.replace("/", "-")))

        if os.path.exists(save_path):
            est_notes = io.load_notes_from_midi(save_path)
            if est_notes is None:
                est_intervals = []
                est_pitches = []
            else:
                est_intervals, est_pitches, _ = est_notes.to_mir_eval()
        else:
            est_intervals, est_pitches = model_inference(audio_path, model, save_path)

        ref_intervals, ref_pitches, _ = note_data.to_mir_eval()

        if len(est_intervals) == 0 or len(ref_intervals) == 0:
            scores_trackid = {}
        else:
            scores_trackid = mir_eval.transcription.evaluate(ref_intervals, ref_pitches, est_intervals, est_pitches)

        scores[track_id] = scores_trackid
        scores[track_id]["instrument"] = instrument

    with open("scores_{}.json".format(model_name), "w") as fhandle:
        json.dump(scores, fhandle)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")
    parser.add_argument(
        "-m",
        "--model",
        type=str,
        help="Which model to run evaluation on",
    )
    parser.add_argument("--data-home", type=str, help="Location to store evaluation data.")
    args = parser.parse_args()

    main(args.model, args.data_home)
