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
import tensorflow as tf
import mir_eval
import librosa

from typing import Dict

from basic_pitch.constants import (
    AUDIO_SAMPLE_RATE,
    ANNOTATIONS_FPS,
    ANNOTATIONS_N_SEMITONES,
    ANNOTATIONS_BASE_FREQUENCY,
    ANNOT_N_FRAMES,
    NOTES_BINS_PER_SEMITONE,
    AUDIO_N_SAMPLES,
)
from basic_pitch import models

SONIFY_FS = 3000
MAX_OUTPUTS = 4

FREQS = librosa.core.cqt_frequencies(
    n_bins=ANNOTATIONS_N_SEMITONES * NOTES_BINS_PER_SEMITONE,
    fmin=ANNOTATIONS_BASE_FREQUENCY,
    bins_per_octave=12 * NOTES_BINS_PER_SEMITONE,
)
# this function is slow - for speed, only sonify frequencies below
# sonify_fs/2 Hz (e.g. 1000 Hz)
MAX_FREQ_INDEX = np.where(FREQS > SONIFY_FS / 2)[0][0]
TIMES = librosa.core.frames_to_time(
    np.arange(ANNOT_N_FRAMES),
    sr=AUDIO_SAMPLE_RATE,
    hop_length=AUDIO_SAMPLE_RATE // ANNOTATIONS_FPS,
)


def get_input_model() -> tf.keras.Model:
    """define a model that generates the CQT (Constant-Q Transform) of input audio"""
    inputs = tf.keras.Input(shape=(AUDIO_N_SAMPLES, 1))  # (batch, time, ch)
    x = models.get_cqt(inputs, 1, False)
    model = tf.keras.Model(inputs=inputs, outputs=x)
    model.compile()
    return model


INPUT_MODEL = get_input_model()


def visualize_transcription(
    file_writer: tf.summary.SummaryWriter,
    stage: str,
    inputs: tf.Tensor,
    targets: Dict[str, tf.Tensor],
    outputs: Dict[str, tf.Tensor],
    loss: float,
    step: int,
    sonify: bool = True,
    contours: bool = True,
) -> None:
    """Create tf.summaries of transcription outputs to be plotted in tensorboard

    Args:
        file_writer: tensorboard filewriter object
        stage: train or validation
        inputs: batch of input data (audio)
        targets: batch of target data (dictionary)
        outputs: batch of output data (dictionary)
        loss: loss value for epoch
        step: which epoch this is
        sonify: sonify outputs
        contours: plot note contours
    """
    with file_writer.as_default():
        # create audio player
        tf.summary.audio(
            f"{stage}/audio/inputs",
            inputs,
            sample_rate=AUDIO_SAMPLE_RATE,
            step=step,
            max_outputs=MAX_OUTPUTS,
        )
        # plot mel spectrograms
        tf.summary.image(
            f"{stage}/audio/input",
            _audio_input(inputs),
            step=step,
            max_outputs=MAX_OUTPUTS,
        )

        # plot onsets
        tf.summary.image(
            f"{stage}/images/onsets/target",
            _array_to_image(targets["onset"]),
            step=step,
            max_outputs=MAX_OUTPUTS,
        )
        tf.summary.image(
            f"{stage}/images/onsets/output",
            _array_to_image(outputs["onset"]),
            step=step,
            max_outputs=MAX_OUTPUTS,
        )

        if sonify:
            tf.summary.audio(
                f"{stage}/audio/onsets-output",
                _array_to_sonification(outputs["onset"], MAX_OUTPUTS),
                sample_rate=SONIFY_FS,
                step=step,
                max_outputs=MAX_OUTPUTS,
            )

        if contours:
            # plot contours
            tf.summary.image(
                f"{stage}/images/contours/target",
                _array_to_image(targets["contour"]),
                step=step,
                max_outputs=MAX_OUTPUTS,
            )
            tf.summary.image(
                f"{stage}/images/contours/output",
                _array_to_image(outputs["contour"]),
                step=step,
                max_outputs=MAX_OUTPUTS,
            )

        # plot notes
        tf.summary.image(
            f"{stage}/images/notes/target",
            _array_to_image(targets["note"]),
            step=step,
            max_outputs=MAX_OUTPUTS,
        )
        tf.summary.image(
            f"{stage}/images/notes/output",
            _array_to_image(outputs["note"]),
            step=step,
            max_outputs=MAX_OUTPUTS,
        )

        if sonify:
            # sonify notes
            tf.summary.audio(
                f"{stage}/audio/notes-output",
                _array_to_sonification(outputs["note"], MAX_OUTPUTS),
                sample_rate=SONIFY_FS,
                step=step,
                max_outputs=MAX_OUTPUTS,
            )

        # plot loss
        tf.summary.scalar(f"{stage}/loss", loss, step=step)
        # plot max
        if contours:
            tf.summary.scalar(f"{stage}/contour-max", np.max(outputs["contour"]), step=step)

        tf.summary.scalar(f"{stage}/note-max", np.max(outputs["note"]), step=step)
        tf.summary.scalar(f"{stage}/onset-max", np.max(outputs["onset"]), step=step)


def _array_to_sonification(array: tf.Tensor, max_outputs: int, clip: float = 0.3) -> tf.Tensor:
    """sonify time frequency representation of audio

    Args:
        array: time-frequency representation of audio
        max_outputs: the number of grams / batches to process / append to the resulting output
        clip: value below which signal is 0'd out.
    """
    gram_batch = tf.transpose(array, perm=[0, 2, 1]).numpy()
    audio_list = []

    for i, gram in enumerate(gram_batch):

        gram[gram < clip] = 0.0
        y = mir_eval.sonify.time_frequency(
            gram[:MAX_FREQ_INDEX, :],
            FREQS[:MAX_FREQ_INDEX],
            TIMES,
            fs=SONIFY_FS,
        )
        audio_list.append(y[:, np.newaxis])
        if i + 1 >= max_outputs:
            break

    return tf.convert_to_tensor(np.array(audio_list), dtype=tf.float32)


def _audio_input(audio: tf.Tensor) -> tf.Tensor:
    """Gets the Constant-Q transform of audio input using the input model defined above.

    Args:
        audio: the audio signal to process

    Returns:
        constant-q transform of the audio (3 bins per semitone, ~11ms hop size.)
    """
    audio_in = INPUT_MODEL(audio)
    return tf.transpose(audio_in, perm=[0, 2, 1, 3])


def _array_to_image(array: tf.Tensor) -> tf.Tensor:
    """Convert a time-frequency array shape=(batch, time, frequency) to
    the shape expected by tf.summary.image (batch, frequency, time, 1)

    Args:
        array: a (batch, time, frequency) array

    Returns:
        reshaped array
    """
    return tf.expand_dims(tf.transpose(array, perm=[0, 2, 1]), 3)
