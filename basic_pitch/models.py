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

from typing import Any, Callable, Dict
import numpy as np
import tensorflow as tf

from basic_pitch import nn
from basic_pitch.constants import (
    ANNOTATIONS_BASE_FREQUENCY,
    ANNOTATIONS_N_SEMITONES,
    AUDIO_N_SAMPLES,
    AUDIO_SAMPLE_RATE,
    CONTOURS_BINS_PER_SEMITONE,
    FFT_HOP,
    N_FREQ_BINS_CONTOURS,
)
from basic_pitch.layers import signal, nnaudio

tfkl = tf.keras.layers

MAX_N_SEMITONES = int(np.floor(12.0 * np.log2(0.5 * AUDIO_SAMPLE_RATE / ANNOTATIONS_BASE_FREQUENCY)))


def transcription_loss(y_true: tf.Tensor, y_pred: tf.Tensor, label_smoothing: float) -> tf.Tensor:
    """Really a binary cross entropy loss. Used to calculate the loss between the predicted
    posteriorgrams and the ground truth matrices.

    Args:
        y_true: The true labels.
        y_pred: The predicted labels.
        label_smoothing: Squeeze labels towards 0.5.

    Returns:
        The transcription loss.
    """
    bce = tf.keras.losses.binary_crossentropy(y_true, y_pred, label_smoothing=label_smoothing)
    return bce


def weighted_transcription_loss(
    y_true: tf.Tensor, y_pred: tf.Tensor, label_smoothing: float, positive_weight: float = 0.5
) -> tf.Tensor:
    """The transcription loss where the positive and negative true labels are balanced by a weighting factor.

    Args:
        y_true: The true labels.
        y_pred: The predicted labels.
        label_smoothing: Smoothing factor. Squeezes labels towards 0.5.
        positive_weight: Weighting factor for the positive labels.

    Returns:
        The weighted transcription loss.
    """
    negative_mask = tf.equal(y_true, 0)
    nonnegative_mask = tf.logical_not(negative_mask)
    bce_negative = tf.keras.losses.binary_crossentropy(
        tf.boolean_mask(y_true, negative_mask),
        tf.boolean_mask(y_pred, negative_mask),
        label_smoothing=label_smoothing,
    )
    bce_nonnegative = tf.keras.losses.binary_crossentropy(
        tf.boolean_mask(y_true, nonnegative_mask),
        tf.boolean_mask(y_pred, nonnegative_mask),
        label_smoothing=label_smoothing,
    )
    return ((1 - positive_weight) * bce_negative) + (positive_weight * bce_nonnegative)


def onset_loss(
    weighted: bool, label_smoothing: float, positive_weight: float
) -> Callable[[tf.Tensor, tf.Tensor], tf.Tensor]:
    """

    Args:
        weighted: Whether or not to use a weighted cross entropy loss.
        label_smoothing: Smoothing factor. Squeezes labels towards 0.5.
        positive_weight: Weighting factor for the positive labels.

    Returns:
        A function that calculates the transcription loss. The function will
        return weighted_transcription_loss if weighted is true else it will return
        transcription_loss.
    """
    if weighted:
        return lambda x, y: weighted_transcription_loss(
            x, y, label_smoothing=label_smoothing, positive_weight=positive_weight
        )
    return lambda x, y: transcription_loss(x, y, label_smoothing=label_smoothing)


def loss(label_smoothing: float = 0.2, weighted: bool = False, positive_weight: float = 0.5) -> Dict[str, Any]:
    """Creates a keras-compatible dictionary of loss functions to calculate
    the loss for the contour, note and onset posteriorgrams.

    Args:
        label_smoothing: Smoothing factor. Squeezes labels towards 0.5.
        weighted: Whether or not to use a weighted cross entropy loss.
        positive_weight: Weighting factor for the positive labels.

    Returns:
        A dictionary with keys "contour," "note," and "onset" with functions as values to be used to calculate
        transcription losses.

    """
    loss_fn = lambda x, y: transcription_loss(x, y, label_smoothing=label_smoothing)
    loss_onset = onset_loss(weighted, label_smoothing, positive_weight)
    return {
        "contour": loss_fn,
        "note": loss_fn,
        "onset": loss_onset,
    }


def _initializer() -> tf.keras.initializers.VarianceScaling:
    return tf.keras.initializers.VarianceScaling(scale=2.0, mode="fan_avg", distribution="uniform", seed=None)


def _kernel_constraint() -> tf.keras.constraints.UnitNorm:
    return tf.keras.constraints.UnitNorm(axis=[0, 1, 2])


def get_cqt(inputs: tf.Tensor, n_harmonics: int, use_batchnorm: bool) -> tf.Tensor:
    """Calculate the CQT of the input audio.

    Input shape: (batch, number of audio samples, 1)
    Output shape: (batch, number of frequency bins, number of time frames)

    Args:
        inputs: The audio input.
        n_harmonics: The number of harmonics to capture above the maximum output frequency.
            Used to calculate the number of semitones for the CQT.
        use_batchnorm: If True, applies batch normalization after computing the CQT

    Returns:
        The log-normalized CQT of the input audio.
    """
    n_semitones = np.min(
        [
            int(np.ceil(12.0 * np.log2(n_harmonics)) + ANNOTATIONS_N_SEMITONES),
            MAX_N_SEMITONES,
        ]
    )
    x = nn.FlattenAudioCh()(inputs)
    x = nnaudio.CQT(
        sr=AUDIO_SAMPLE_RATE,
        hop_length=FFT_HOP,
        fmin=ANNOTATIONS_BASE_FREQUENCY,
        n_bins=n_semitones * CONTOURS_BINS_PER_SEMITONE,
        bins_per_octave=12 * CONTOURS_BINS_PER_SEMITONE,
    )(x)
    x = signal.NormalizedLog()(x)
    x = tf.expand_dims(x, -1)
    if use_batchnorm:
        x = tfkl.BatchNormalization()(x)
    return x


def model(
    n_harmonics: int = 8,
    n_filters_contour: int = 32,
    n_filters_onsets: int = 32,
    n_filters_notes: int = 32,
    no_contours: bool = False,
) -> tf.keras.Model:
    """Basic Pitch's model implementation.

    Args:
        n_harmonics: The number of harmonics to use in the harmonic stacking layer.
        n_filters_contour: Number of filters for the contour convolutional layer.
        n_filters_onsets: Number of filters for the onsets convolutional layer.
        n_filters_notes: Number of filters for the notes convolutional layer.
        no_contours: Whether or not to include contours in the output.
    """
    # input representation
    inputs = tf.keras.Input(shape=(AUDIO_N_SAMPLES, 1))  # (batch, time, ch)
    x = get_cqt(inputs, n_harmonics, True)

    if n_harmonics > 1:
        x = nn.HarmonicStacking(
            CONTOURS_BINS_PER_SEMITONE,
            [0.5] + list(range(1, n_harmonics)),
            N_FREQ_BINS_CONTOURS,
        )(x)
    else:
        x = nn.HarmonicStacking(
            CONTOURS_BINS_PER_SEMITONE,
            [1],
            N_FREQ_BINS_CONTOURS,
        )(x)

    # contour layers - fully convolutional
    x_contours = tfkl.Conv2D(
        n_filters_contour,
        (5, 5),
        padding="same",
        kernel_initializer=_initializer(),
        kernel_constraint=_kernel_constraint(),
    )(x)

    x_contours = tfkl.BatchNormalization()(x_contours)
    x_contours = tfkl.ReLU()(x_contours)

    x_contours = tfkl.Conv2D(
        8,
        (3, 3 * 13),
        padding="same",
        kernel_initializer=_initializer(),
        kernel_constraint=_kernel_constraint(),
    )(x)

    x_contours = tfkl.BatchNormalization()(x_contours)
    x_contours = tfkl.ReLU()(x_contours)

    if not no_contours:
        contour_name = "contour"
        x_contours = tfkl.Conv2D(
            1,
            (5, 5),
            padding="same",
            activation="sigmoid",
            kernel_initializer=_initializer(),
            kernel_constraint=_kernel_constraint(),
            name="contours-reduced",
        )(x_contours)
        x_contours = nn.FlattenFreqCh(name=contour_name)(x_contours)  # contour output

        # reduced contour output as input to notes
        x_contours_reduced = tf.expand_dims(x_contours, -1)
    else:
        x_contours_reduced = x_contours

    x_contours_reduced = tfkl.Conv2D(
        n_filters_notes,
        (7, 7),
        padding="same",
        strides=(1, 3),
        kernel_initializer=_initializer(),
        kernel_constraint=_kernel_constraint(),
    )(x_contours_reduced)
    x_contours_reduced = tfkl.ReLU()(x_contours_reduced)

    # note output layer
    note_name = "note"
    x_notes_pre = tfkl.Conv2D(
        1,
        (7, 3),
        padding="same",
        kernel_initializer=_initializer(),
        kernel_constraint=_kernel_constraint(),
        activation="sigmoid",
    )(x_contours_reduced)
    x_notes = nn.FlattenFreqCh(name=note_name)(x_notes_pre)

    # onset output layer

    # onsets - fully convolutional
    x_onset = tfkl.Conv2D(
        n_filters_onsets,
        (5, 5),
        padding="same",
        strides=(1, 3),
        kernel_initializer=_initializer(),
        kernel_constraint=_kernel_constraint(),
    )(x)
    x_onset = tfkl.BatchNormalization()(x_onset)
    x_onset = tfkl.ReLU()(x_onset)
    x_onset = tfkl.Concatenate(axis=3, name="concat")([x_notes_pre, x_onset])
    x_onset = tfkl.Conv2D(
        1,
        (3, 3),
        padding="same",
        activation="sigmoid",
        kernel_initializer=_initializer(),
        kernel_constraint=_kernel_constraint(),
    )(x_onset)

    onset_name = "onset"
    x_onset = nn.FlattenFreqCh(
        name=onset_name,
    )(x_onset)

    outputs = {"onset": x_onset, "contour": x_contours, "note": x_notes}

    return tf.keras.Model(inputs=inputs, outputs=outputs)
