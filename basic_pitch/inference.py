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
import enum
import json
import logging
import os
import pathlib
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union, cast


from basic_pitch import CT_PRESENT, ONNX_PRESENT, TF_PRESENT, TFLITE_PRESENT

try:
    import tensorflow as tf
except ImportError:
    pass

try:
    import coremltools as ct
except ImportError:
    pass

try:
    import tflite_runtime.interpreter as tflite
except ImportError:
    if TF_PRESENT:
        import tensorflow.lite as tflite

try:
    import onnxruntime as ort
except ImportError:
    pass

import numpy as np
import numpy.typing as npt
import librosa
import pretty_midi

from basic_pitch.constants import (
    AUDIO_SAMPLE_RATE,
    AUDIO_N_SAMPLES,
    ANNOTATIONS_FPS,
    FFT_HOP,
)
from basic_pitch.commandline_printing import (
    generating_file_message,
    no_tf_warnings,
    file_saved_confirmation,
    failed_to_save,
)
import basic_pitch.note_creation as infer


class Model:
    class MODEL_TYPES(enum.Enum):
        TENSORFLOW = enum.auto()
        COREML = enum.auto()
        TFLITE = enum.auto()
        ONNX = enum.auto()

    def __init__(self, model_path: Union[pathlib.Path, str]):
        present = []
        if TF_PRESENT:
            present.append("TensorFlow")
            try:
                self.model_type = Model.MODEL_TYPES.TENSORFLOW
                self.model = tf.saved_model.load(str(model_path))
                return
            except Exception as e:
                if os.path.isdir(model_path) and {"saved_model.pb", "variables"} & set(os.listdir(model_path)):
                    logging.warning(
                        "Could not load TensorFlow saved model %s even "
                        "though it looks like a saved model file with error %s. "
                        "Are you sure it's a TensorFlow saved model?",
                        model_path,
                        e.__repr__(),
                    )

        if CT_PRESENT:
            present.append("CoreML")
            try:
                self.model_type = Model.MODEL_TYPES.COREML
                self.model = ct.models.MLModel(str(model_path))
                return
            except Exception as e:
                if str(model_path).endswith(".mlpackage"):
                    logging.warning(
                        "Could not load CoreML file %s even "
                        "though it looks like a CoreML file with error %s. "
                        "Are you sure it's a CoreML file?",
                        model_path,
                        e.__repr__(),
                    )

        if TFLITE_PRESENT or TF_PRESENT:
            present.append("TensorFlowLite")
            try:
                self.model_type = Model.MODEL_TYPES.TFLITE
                self.interpreter = tflite.Interpreter(str(model_path))
                self.model = self.interpreter.get_signature_runner()
                return
            except Exception as e:
                if str(model_path).endswith(".tflite"):
                    logging.warning(
                        "Could not load TensorFlowLite file %s even "
                        "though it looks like a TFLite file with error %s. "
                        "Are you sure it's a TFLite file?",
                        model_path,
                        e.__repr__(),
                    )

        if ONNX_PRESENT:
            present.append("ONNX")
            try:
                self.model_type = Model.MODEL_TYPES.ONNX
                self.model = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
                return
            except Exception as e:
                if str(model_path).endswith(".onnx"):
                    logging.warning(
                        "Could not load ONNX file %s even "
                        "though it looks like a ONNX file with error %s. "
                        "Are you sure it's a ONNX file?",
                        model_path,
                        e.__repr__(),
                    )

        raise ValueError(
            f"File {model_path} cannot be loaded into either "
            "TensorFlow, CoreML, TFLite or ONNX. "
            "Please check if it is a supported and valid serialized model "
            "and that one of these packages are installed. On this system, "
            f"{present} is installed."
        )

    def predict(self, x: npt.NDArray[np.float32]) -> Dict[str, npt.NDArray[np.float32]]:
        if self.model_type == Model.MODEL_TYPES.TENSORFLOW:
            return {k: v.numpy() for k, v in cast(tf.keras.Model, self.model(x)).items()}
        elif self.model_type == Model.MODEL_TYPES.COREML:
            print(f"isfinite: {np.all(np.isfinite(x))}", flush=True)
            print(f"shape: {x.shape}", flush=True)
            print(f"dtype: {x.dtype}", flush=True)
            result = cast(ct.models.MLModel, self.model).predict({"input_2": x})
            return {
                "note": result["Identity_1"],
                "onset": result["Identity_2"],
                "contour": result["Identity"],
            }
        elif self.model_type == Model.MODEL_TYPES.TFLITE:
            return self.model(input_2=x)  # type: ignore
        elif self.model_type == Model.MODEL_TYPES.ONNX:
            return {
                k: v
                for k, v in zip(
                    ["note", "onset", "contour"],
                    cast(ort.InferenceSession, self.model).run(
                        [
                            "StatefulPartitionedCall:1",
                            "StatefulPartitionedCall:2",
                            "StatefulPartitionedCall:0",
                        ],
                        {"serving_default_input_2:0": x},
                    ),
                )
            }


def window_audio_file(
    audio_original: npt.NDArray[np.float32], hop_size: int
) -> Iterable[Tuple[npt.NDArray[np.float32], Dict[str, float]]]:
    """
    Pad appropriately an audio file, and return as
    windowed signal, with window length = AUDIO_N_SAMPLES

    Returns:
        audio_windowed: tensor with shape (n_windows, AUDIO_N_SAMPLES, 1)
            audio windowed into fixed length chunks
        window_times: list of {'start':.., 'end':...} objects (times in seconds)

    """
    for i in range(0, audio_original.shape[0], hop_size):
        window = audio_original[i : i + AUDIO_N_SAMPLES]
        if len(window) < AUDIO_N_SAMPLES:
            window = np.pad(
                window,
                pad_width=[[0, AUDIO_N_SAMPLES - len(window)]],
            )
        t_start = float(i) / AUDIO_SAMPLE_RATE
        window_time = {
            "start": t_start,
            "end": t_start + (AUDIO_N_SAMPLES / AUDIO_SAMPLE_RATE),
        }
        yield np.expand_dims(window, axis=-1), window_time


def get_audio_input(
    audio_path: Union[pathlib.Path, str], overlap_len: int, hop_size: int
) -> Iterable[Tuple[npt.NDArray[np.float32], Dict[str, float], int]]:
    """
    Read wave file (as mono), pad appropriately, and return as
    windowed signal, with window length = AUDIO_N_SAMPLES

    Returns:
        audio_windowed: tensor with shape (n_windows, AUDIO_N_SAMPLES, 1)
            audio windowed into fixed length chunks
        window_times: list of {'start':.., 'end':...} objects (times in seconds)
        audio_original_length: int
            length of original audio file, in frames, BEFORE padding.

    """
    assert overlap_len % 2 == 0, "overlap_length must be even, got {}".format(overlap_len)

    audio_original, _ = librosa.load(str(audio_path), sr=AUDIO_SAMPLE_RATE, mono=True)

    original_length = audio_original.shape[0]
    audio_original = np.concatenate([np.zeros((int(overlap_len / 2),), dtype=np.float32), audio_original])
    for window, window_time in window_audio_file(audio_original, hop_size):
        yield np.expand_dims(window, axis=0), window_time, original_length


def unwrap_output(
    output: npt.NDArray[np.float32],
    audio_original_length: int,
    n_overlapping_frames: int,
) -> np.array:
    """Unwrap batched model predictions to a single matrix.

    Args:
        output: array (n_batches, n_times_short, n_freqs)
        audio_original_length: length of original audio signal (in samples)
        n_overlapping_frames: number of overlapping frames in the output

    Returns:
        array (n_times, n_freqs)
    """
    if len(output.shape) != 3:
        return None

    n_olap = int(0.5 * n_overlapping_frames)
    if n_olap > 0:
        # remove half of the overlapping frames from beginning and end
        output = output[:, n_olap:-n_olap, :]

    output_shape = output.shape
    n_output_frames_original = int(np.floor(audio_original_length * (ANNOTATIONS_FPS / AUDIO_SAMPLE_RATE)))
    unwrapped_output = output.reshape(output_shape[0] * output_shape[1], output_shape[2])
    return unwrapped_output[:n_output_frames_original, :]  # trim to original audio length


def run_inference(
    audio_path: Union[pathlib.Path, str],
    model_or_model_path: Union[Model, pathlib.Path, str],
    debug_file: Optional[pathlib.Path] = None,
) -> Dict[str, np.array]:
    """Run the model on the input audio path.

    Args:
        audio_path: The audio to run inference on.
        model_or_model_path: A loaded Model or path to a serialized model to load.
        debug_file: An optional path to output debug data to. Useful for testing/verification.

    Returns:
       A dictionary with the notes, onsets and contours from model inference.
    """
    if isinstance(model_or_model_path, Model):
        model = model_or_model_path
    else:
        model = Model(model_or_model_path)

    # overlap 30 frames
    n_overlapping_frames = 30
    overlap_len = n_overlapping_frames * FFT_HOP
    hop_size = AUDIO_N_SAMPLES - overlap_len

    output: Dict[str, Any] = {"note": [], "onset": [], "contour": []}
    for audio_windowed, _, audio_original_length in get_audio_input(audio_path, overlap_len, hop_size):
        for k, v in model.predict(audio_windowed).items():
            output[k].append(v)

    unwrapped_output = {
        k: unwrap_output(np.concatenate(output[k]), audio_original_length, n_overlapping_frames) for k in output
    }

    if debug_file:
        with open(debug_file, "w") as f:
            json.dump(
                {
                    "audio_windowed": audio_windowed.numpy().tolist(),
                    "audio_original_length": audio_original_length,
                    "hop_size_samples": hop_size,
                    "overlap_length_samples": overlap_len,
                    "unwrapped_output": {k: v.tolist() for k, v in unwrapped_output.items()},
                },
                f,
            )

    return unwrapped_output


class OutputExtensions(enum.Enum):
    MIDI = "mid"
    MODEL_OUTPUT_NPZ = "npz"
    MIDI_SONIFICATION = "wav"
    NOTE_EVENTS = "csv"


def verify_input_path(audio_path: Union[pathlib.Path, str]) -> None:
    """Verify that an input path is valid and can be processed

    Args:
        audio_path: Path to an audio file.

    Raises:
        ValueError: If the audio file is invalid.
    """
    if not os.path.isfile(audio_path):
        raise ValueError(f"🚨 {audio_path} is not a file path.")

    if not os.path.exists(audio_path):
        raise ValueError(f"🚨 {audio_path} does not exist.")


def verify_output_dir(output_dir: Union[pathlib.Path, str]) -> None:
    """Verify that an output directory is valid and can be processed

    Args:
        output_dir: Path to an output directory.

    Raises:
        ValueError: If the output directory is invalid.
    """
    if not os.path.isdir(output_dir):
        raise ValueError(f"🚨 {output_dir} is not a directory.")

    if not os.path.exists(output_dir):
        raise ValueError(f"🚨 {output_dir} does not exist.")


def build_output_path(
    audio_path: Union[pathlib.Path, str],
    output_directory: Union[pathlib.Path, str],
    output_type: OutputExtensions,
) -> pathlib.Path:
    """Create an output path and make sure it doesn't already exist.

    Args:
        audio_path: The original file path.
        output_directory: The directory we will output to.
        output_type: The type of output file we are creating.

    Raises:
        IOError: If the generated path already exists.

    Returns:
        A new path in the output_directory with the stem audio_path and an extension
        based on output_type.
    """
    audio_path = str(audio_path)
    if not isinstance(output_directory, pathlib.Path):
        output_directory = pathlib.Path(output_directory)

    basename, _ = os.path.splitext(os.path.basename(audio_path))

    output_path = output_directory / f"{basename}_basic_pitch.{output_type.value}"

    generating_file_message(output_type.name)

    if output_path.exists():
        raise IOError(
            f"  🚨 {str(output_path)} already exists and would be overwritten. Skipping output files for {audio_path}."
        )

    return output_path


def save_note_events(
    note_events: List[Tuple[float, float, int, float, Optional[List[int]]]],
    save_path: Union[pathlib.Path, str],
) -> None:
    """Save note events to file

    Args:
        note_events: A list of note event tuples to save. Tuples have the format
            ("start_time_s", "end_time_s", "pitch_midi", "velocity", "list of pitch bend values")
        save_path: The location we're saving it
    """

    with open(save_path, "w") as fhandle:
        writer = csv.writer(fhandle, delimiter=",")
        writer.writerow(["start_time_s", "end_time_s", "pitch_midi", "velocity", "pitch_bend"])
        for start_time, end_time, note_number, amplitude, pitch_bend in note_events:
            row = [start_time, end_time, note_number, int(np.round(127 * amplitude))]
            if pitch_bend:
                row.extend(pitch_bend)
            writer.writerow(row)


def predict(
    audio_path: Union[pathlib.Path, str],
    model_or_model_path: Union[Model, pathlib.Path, str],
    onset_threshold: float = 0.5,
    frame_threshold: float = 0.3,
    minimum_note_length: float = 127.70,
    minimum_frequency: Optional[float] = None,
    maximum_frequency: Optional[float] = None,
    multiple_pitch_bends: bool = False,
    melodia_trick: bool = True,
    debug_file: Optional[pathlib.Path] = None,
    midi_tempo: float = 120,
) -> Tuple[
    Dict[str, np.array],
    pretty_midi.PrettyMIDI,
    List[Tuple[float, float, int, float, Optional[List[int]]]],
]:
    """Run a single prediction.

    Args:
        audio_path: File path for the audio to run inference on.
        model_or_model_path: A loaded Model or path to a serialized model to load.
        onset_threshold: Minimum energy required for an onset to be considered present.
        frame_threshold: Minimum energy requirement for a frame to be considered present.
        minimum_note_length: The minimum allowed note length in milliseconds.
        minimum_freq: Minimum allowed output frequency, in Hz. If None, all frequencies are used.
        maximum_freq: Maximum allowed output frequency, in Hz. If None, all frequencies are used.
        multiple_pitch_bends: If True, allow overlapping notes in midi file to have pitch bends.
        melodia_trick: Use the melodia post-processing step.
        debug_file: An optional path to output debug data to. Useful for testing/verification.
    Returns:
        The model output, midi data and note events from a single prediction
    """

    with no_tf_warnings():
        print(f"Predicting MIDI for {audio_path}...")

        model_output = run_inference(audio_path, model_or_model_path, debug_file)
        min_note_len = int(np.round(minimum_note_length / 1000 * (AUDIO_SAMPLE_RATE / FFT_HOP)))
        midi_data, note_events = infer.model_output_to_notes(
            model_output,
            onset_thresh=onset_threshold,
            frame_thresh=frame_threshold,
            min_note_len=min_note_len,  # convert to frames
            min_freq=minimum_frequency,
            max_freq=maximum_frequency,
            multiple_pitch_bends=multiple_pitch_bends,
            melodia_trick=melodia_trick,
            midi_tempo=midi_tempo,
        )

    if debug_file:
        with open(debug_file) as f:
            debug_data = json.load(f)
        with open(debug_file, "w") as f:
            json.dump(
                {
                    **debug_data,
                    "min_note_length": min_note_len,
                    "onset_thresh": onset_threshold,
                    "frame_thresh": frame_threshold,
                    "estimated_notes": [
                        (
                            float(start_time),
                            float(end_time),
                            int(pitch),
                            float(amplitude),
                            [int(b) for b in pitch_bends] if pitch_bends else None,
                        )
                        for start_time, end_time, pitch, amplitude, pitch_bends in note_events
                    ],
                },
                f,
            )

    return model_output, midi_data, note_events


def predict_and_save(
    audio_path_list: Sequence[Union[pathlib.Path, str]],
    output_directory: Union[pathlib.Path, str],
    save_midi: bool,
    sonify_midi: bool,
    save_model_outputs: bool,
    save_notes: bool,
    model_or_model_path: Union[Model, str, pathlib.Path],
    onset_threshold: float = 0.5,
    frame_threshold: float = 0.3,
    minimum_note_length: float = 127.70,
    minimum_frequency: Optional[float] = None,
    maximum_frequency: Optional[float] = None,
    multiple_pitch_bends: bool = False,
    melodia_trick: bool = True,
    debug_file: Optional[pathlib.Path] = None,
    sonification_samplerate: int = 44100,
    midi_tempo: float = 120,
) -> None:
    """Make a prediction and save the results to file.

    Args:
        audio_path_list: List of file paths for the audio to run inference on.
        output_directory: Directory to output MIDI and all other outputs derived from the model to.
        save_midi: True to save midi.
        sonify_midi: Whether or not to render audio from the MIDI and output it to a file.
        save_model_outputs: True to save contours, onsets and notes from the model prediction.
        save_notes: True to save note events.
        model_or_model_path: A loaded Model or path to a serialized model to load.
        onset_threshold: Minimum energy required for an onset to be considered present.
        frame_threshold: Minimum energy requirement for a frame to be considered present.
        minimum_note_length: The minimum allowed note length in milliseconds.
        minimum_freq: Minimum allowed output frequency, in Hz. If None, all frequencies are used.
        maximum_freq: Maximum allowed output frequency, in Hz. If None, all frequencies are used.
        multiple_pitch_bends: If True, allow overlapping notes in midi file to have pitch bends.
        melodia_trick: Use the melodia post-processing step.
        debug_file: An optional path to output debug data to. Useful for testing/verification.
        sonification_samplerate: Sample rate for rendering audio from MIDI.
    """
    for audio_path in audio_path_list:
        print("")
        try:
            model_output, midi_data, note_events = predict(
                pathlib.Path(audio_path),
                model_or_model_path,
                onset_threshold,
                frame_threshold,
                minimum_note_length,
                minimum_frequency,
                maximum_frequency,
                multiple_pitch_bends,
                melodia_trick,
                debug_file,
                midi_tempo,
            )

            if save_model_outputs:
                model_output_path = build_output_path(audio_path, output_directory, OutputExtensions.MODEL_OUTPUT_NPZ)
                try:
                    np.savez(model_output_path, basic_pitch_model_output=model_output)
                    file_saved_confirmation(OutputExtensions.MODEL_OUTPUT_NPZ.name, model_output_path)
                except Exception as e:
                    failed_to_save(OutputExtensions.MODEL_OUTPUT_NPZ.name, model_output_path)
                    raise e

            if save_midi:
                try:
                    midi_path = build_output_path(audio_path, output_directory, OutputExtensions.MIDI)
                except IOError as e:
                    raise e
                try:
                    midi_data.write(str(midi_path))
                    file_saved_confirmation(OutputExtensions.MIDI.name, midi_path)
                except Exception as e:
                    failed_to_save(OutputExtensions.MIDI.name, midi_path)
                    raise e

            if sonify_midi:
                midi_sonify_path = build_output_path(audio_path, output_directory, OutputExtensions.MIDI_SONIFICATION)
                try:
                    infer.sonify_midi(midi_data, midi_sonify_path, sr=sonification_samplerate)
                    file_saved_confirmation(OutputExtensions.MIDI_SONIFICATION.name, midi_sonify_path)
                except Exception as e:
                    failed_to_save(OutputExtensions.MIDI_SONIFICATION.name, midi_sonify_path)
                    raise e

            if save_notes:
                note_events_path = build_output_path(audio_path, output_directory, OutputExtensions.NOTE_EVENTS)
                try:
                    save_note_events(note_events, note_events_path)
                    file_saved_confirmation(OutputExtensions.NOTE_EVENTS.name, note_events_path)
                except Exception as e:
                    failed_to_save(OutputExtensions.NOTE_EVENTS.name, note_events_path)
                    raise e
        except Exception as e:
            raise e
