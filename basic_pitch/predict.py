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
import os
import pathlib
import traceback

from basic_pitch import ICASSP_2022_MODEL_PATH


os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


def main() -> None:
    """Handle command line arguments. Entrypoint for this script."""
    parser = argparse.ArgumentParser(description="Predict midi from audio.")
    parser.add_argument("output_dir", type=str, help="directory to save outputs")
    parser.add_argument("audio_paths", type=str, nargs="+", help="Space separated paths to the input audio files.")
    parser.add_argument(
        "--model_path",
        type=str,
        default=ICASSP_2022_MODEL_PATH,
        help="path to the saved model directory. Defaults to a ICASSP 2022 model",
    )
    parser.add_argument(
        "--save-midi",
        action="store_true",
        default=True,
        help="Create a MIDI file.",
    )
    parser.add_argument(
        "--sonify-midi",
        action="store_true",
        help="Create an audio .wav file which sonifies the MIDI outputs.",
    )
    parser.add_argument(
        "--save-model-outputs",
        action="store_true",
        help="Save the raw model output as an npz file.",
    )
    parser.add_argument(
        "--save-note-events",
        action="store_true",
        help="Save the predicted note events as a csv file.",
    )
    parser.add_argument(
        "--onset-threshold",
        type=float,
        default=0.5,
        help="The minimum likelihood for an onset to occur, between 0 and 1.",
    )
    parser.add_argument(
        "--frame-threshold",
        type=float,
        default=0.3,
        help="The minimum likelihood for a frame to sustain, between 0 and 1.",
    )
    parser.add_argument(
        "--minimum-note-length",
        type=float,
        default=127.70,
        help="The minimum allowed note length, in miliseconds.",
    )
    parser.add_argument(
        "--minimum-frequency",
        type=float,
        default=None,
        help="The minimum allowed note frequency, in Hz.",
    )
    parser.add_argument(
        "--maximum-frequency",
        type=float,
        default=None,
        help="The maximum allowed note frequency, in Hz.",
    )
    parser.add_argument(
        "--multiple-pitch-bends",
        action="store_true",
        help="Allow overlapping notes in midi file to have pitch bends. Note: this will map each "
        "pitch to its own instrument",
    )
    parser.add_argument(
        "--sonification-samplerate",
        type=int,
        default=44100,
        help="The samplerate for sonified audio files.",
    )
    parser.add_argument(
        "--midi-tempo",
        type=float,
        default=120,
        help="The tempo for the midi file.",
    )
    parser.add_argument("--debug-file", default=None, help="Optional file for debug output for inference.")
    parser.add_argument("--no-melodia", default=False, action="store_true", help="Skip the melodia trick.")
    args = parser.parse_args()

    print("")
    print("✨✨✨✨✨✨✨✨✨")
    print("✨ Basic Pitch  ✨")
    print("✨✨✨✨✨✨✨✨✨")
    print("")

    # tensorflow is very slow to import
    # this import is here so that the help messages print faster
    print("Importing Tensorflow (this may take a few seconds)...")
    from basic_pitch.inference import predict_and_save, verify_output_dir, verify_input_path

    output_dir = pathlib.Path(args.output_dir)
    verify_output_dir(output_dir)

    audio_path_list = [pathlib.Path(audio_path) for audio_path in args.audio_paths]
    for audio_path in audio_path_list:
        verify_input_path(audio_path)

    try:
        predict_and_save(
            audio_path_list,
            output_dir,
            args.save_midi,
            args.sonify_midi,
            args.save_model_outputs,
            args.save_note_events,
            pathlib.Path(args.model_path),
            args.onset_threshold,
            args.frame_threshold,
            args.minimum_note_length,
            args.minimum_frequency,
            args.maximum_frequency,
            args.multiple_pitch_bends,
            not args.no_melodia,
            pathlib.Path(args.debug_file) if args.debug_file else None,
            args.sonification_samplerate,
            args.midi_tempo,
        )
        print("\n✨ Done ✨\n")
    except IOError as ioe:
        print(ioe)
    except Exception as e:
        print("🚨 Something went wrong 😔 - see the traceback below for details.")
        print("")
        print(e)
        print(traceback.format_exc())


if __name__ == "__main__":
    main()
