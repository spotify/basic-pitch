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

import os
import uuid
from typing import Dict, List, Tuple

import numpy as np
import tensorflow as tf

# import tensorflow_addons as tfa

from basic_pitch.constants import (
    ANNOTATIONS_FPS,
    ANNOT_N_FRAMES,
    AUDIO_N_CHANNELS,
    AUDIO_N_SAMPLES,
    AUDIO_SAMPLE_RATE,
    AUDIO_WINDOW_LENGTH,
    N_FREQ_BINS_NOTES,
    N_FREQ_BINS_CONTOURS,
)

N_SAMPLES_PER_TRACK = 20


def prepare_datasets(
    datasets_base_path,
    training_shuffle_buffer_size,
    batch_size,
    validation_steps,
    datasets_to_use: List[str],
    dataset_sampling_frequency: np.ndarray,
):
    """
    Return a training and a testing dataset.

    Args:
        training_shuffle_buffer_size : size of shuffle buffer (only for training set)
        batch_size : ..
    """
    assert batch_size > 0
    assert validation_steps is not None and validation_steps > 0
    assert training_shuffle_buffer_size is not None

    # init both
    ds_train = sample_datasets(
        "train",
        datasets_base_path,
        datasets=datasets_to_use,
        dataset_sampling_frequency=dataset_sampling_frequency,
    )
    ds_validation = sample_datasets(
        "validation",
        datasets_base_path,
        datasets=datasets_to_use,
        dataset_sampling_frequency=dataset_sampling_frequency,
    )

    # check that the base dataset returned by ds_function is FINITE
    for ds in [ds_train, ds_validation]:
        tf.debugging.assert_none_equal(
            tf.cast(tf.data.experimental.cardinality(ds), tf.int32),
            tf.data.experimental.INFINITE_CARDINALITY,
        )

    # training dataset
    if training_shuffle_buffer_size > 0:
        # Lets try to cache before the shuffle. This is the entire training dataset so we'll cache
        # to memory
        ds_train = (
            ds_train.shuffle(training_shuffle_buffer_size, reshuffle_each_iteration=True)
            .repeat()
            .batch(batch_size)
            .prefetch(tf.data.AUTOTUNE)
        )

    # validation dataset
    ds_validation = (
        ds_validation.repeat()
        .batch(batch_size)
        .take(validation_steps)
        .cache(f"validation_set_cache_{str(uuid.uuid4())}")
        .repeat()
        .prefetch(tf.data.AUTOTUNE)
    )

    return ds_train, ds_validation


def prepare_visualization_datasets(
    datasets_base_path,
    batch_size,
    validation_steps,
    datasets_to_use: List[str],
    dataset_sampling_frequency: np.ndarray,
):
    """
    Return a training and a testing dataset.

    Args:
        training_shuffle_buffer_size : size of shuffle buffer (only for training set)
        batch_size : ..
    """

    assert batch_size > 0
    assert validation_steps is not None and validation_steps > 0

    ds_train = sample_datasets(
        "train",
        datasets_base_path,
        datasets=datasets_to_use,
        dataset_sampling_frequency=dataset_sampling_frequency,
        n_samples_per_track=1,
    )
    ds_validation = sample_datasets(
        "validation",
        datasets_base_path,
        datasets=datasets_to_use,
        dataset_sampling_frequency=dataset_sampling_frequency,
        n_samples_per_track=1,
    )

    # check that the base dataset returned by ds_function is FINITE
    for ds in [ds_train, ds_validation]:
        tf.debugging.assert_none_equal(
            tf.cast(tf.data.experimental.cardinality(ds), tf.int32),
            tf.data.experimental.INFINITE_CARDINALITY,
        )

    # training dataset
    ds_train = ds_train.repeat().batch(batch_size).prefetch(tf.data.AUTOTUNE)

    # validation dataset
    ds_validation = (
        ds_validation.repeat()
        .batch(batch_size)
        .take(validation_steps)
        .cache(f"validation_set_cache_{str(uuid.uuid4())}")
        .repeat()
        .prefetch(tf.data.AUTOTUNE)
    )

    return ds_train, ds_validation


def sample_datasets(
    split,
    datasets_base_path,
    datasets,
    dataset_sampling_frequency=None,
    n_shuffle=1000,
    n_samples_per_track=N_SAMPLES_PER_TRACK,
    pairs=False,
    num_parallel_calls=6,
):
    assert split in ["train", "validation"]
    if split == "validation":
        n_shuffle = 0
        pairs = False
        if n_samples_per_track != 1:
            n_samples_per_track = 5

    ds_list = []


    file_generator, random_seed = transcription_file_generator(
        split,
        datasets,
        datasets_base_path,
        dataset_sampling_frequency,
    )

    ds_dataset = transcription_dataset(file_generator, n_samples_per_track, random_seed)

    if n_shuffle > 0:
        ds_dataset = ds_dataset.shuffle(n_shuffle)
    ds_list.append(ds_dataset)

    if pairs:
        pairs_generator, random_seed_pairs = transcription_file_generator(
            split,
            datasets,
            datasets_base_path,
            dataset_sampling_frequency,
        )
        pairs_ds = transcription_dataset(
            pairs_generator,
            n_samples_per_track,
            random_seed_pairs,
        )

        pairs_ds = pairs_ds.shuffle(n_samples_per_track * 10)  # shuffle so that different tracks get mixed together
        pairs_ds = pairs_ds.batch(2)
        pairs_ds = pairs_ds.map(combine_transcription_examples)
        ds_list.append(pairs_ds)

    n_datasets = len(ds_list)
    choice_dataset = tf.data.Dataset.range(
        n_datasets
    ).repeat()  # this repeat is critical! if not, only n_dataset points will be sampled!!
    return tf.data.experimental.choose_from_datasets(ds_list, choice_dataset)


def transcription_file_generator(
    split,
    dataset_names,
    datasets_base_path,
    sample_weights,
):
    """
    dataset_names: list of dataset dataset_names
    """
    file_dict = {
        dataset_name: tf.data.Dataset.list_files(
            os.path.join(datasets_base_path, dataset_name, "splits", split, "*tfrecord")
        )
        for dataset_name in dataset_names
    }

    if split == "train":
        return lambda: _train_file_generator(file_dict, sample_weights), False
    return lambda: _validation_file_generator(file_dict), True


def _train_file_generator(x, weights):
    x = {k: list(v) for (k, v) in x.items()}
    keys = list(x.keys())
    # shuffle each list
    for k in keys:
        np.random.shuffle(x[k])

    while all(x.values()):
        # choose a random dataset and yield the last file
        fpath = x[np.random.choice(keys, p=weights)].pop()
        yield fpath


def _validation_file_generator(x):
    x = {k: list(v) for (k, v) in x.items()}
    # loop until there are no more test files
    while any(x.values()):
        # alternate between datasets (dataset 1 elt 1, dataset 2, elt 1, ...)
        # this is so test files in the tensorboard have 4 different datasets
        # instead of 4 elements from 1
        for k in x:
            # if the list of files for this dataset is empty skip it
            if x[k]:
                yield x[k].pop()


def combine_transcription_examples(a, target, w):
    return (
        # mix the audio snippets
        tf.math.reduce_mean(a, axis=0),
        # annotations are the max per bin - active frames stay active
        {
            "onset": tf.math.reduce_max(target["onset"], axis=0),
            "contour": tf.math.reduce_max(target["contour"], axis=0),
            "note": tf.math.reduce_max(target["note"], axis=0),
        },
        # weights are the minimum - if an annotation is missing in one, we should set the weights to zero
        {
            "onset": tf.math.reduce_min(w["onset"], axis=0),
            "contour": tf.math.reduce_min(w["contour"], axis=0),
            "note": tf.math.reduce_min(w["note"], axis=0),
        },
    )


def transcription_dataset(file_generator, n_samples_per_track, random_seed):
    """
    `fpaths_in` is a list of .tfrecords files
    return a tf.Dataset with the following fields (as tuple):
        - audio (shape AUDIO_N_SAMPLES, 1)
        - {'contours': contours, 'notes': notes, 'onsets': onsets}
    contours has shape (ANNOT_N_FRAMES, N_FREQ_BINS_CONTOURS)
    notes and onsets have shape: (ANNOT_N_FRAMES, N_FREQ_BINS_NOTES)
    """
    ds = tf.data.Dataset.from_generator(file_generator, output_types=tf.string, output_shapes=())
    ds = tf.data.TFRecordDataset(ds)
    ds = ds.map(parse_transcription_tfexample, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.filter(is_not_bad_shape)
    ds = ds.map(
        lambda file_id, source, audio_wav, notes_indices, notes_values, onsets_indices, onsets_values, contours_indices, contours_values, notes_onsets_shape, contours_shape: (  # noqa: E501
            file_id,
            source,
            tf.audio.decode_wav(
                audio_wav,
                desired_channels=AUDIO_N_CHANNELS,
                desired_samples=-1,
                name=None,
            ),
            sparse2dense(notes_values, notes_indices, notes_onsets_shape),
            sparse2dense(onsets_values, onsets_indices, notes_onsets_shape),
            sparse2dense(contours_values, contours_indices, contours_shape),
        ),
        num_parallel_calls=tf.data.AUTOTUNE,
    )
    ds = ds.map(reduce_transcription_inputs)
    ds = ds.map(get_sample_weights, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.flat_map(
        lambda a, o, c, n, ow, cw, nw, m: get_transcription_chunks(
            a, o, c, n, ow, cw, nw, n_samples_per_track, random_seed
        )
    )
    ds = ds.filter(is_not_all_silent_annotations)  # remove examples where all annotations are zero
    ds = ds.map(to_transcription_training_input)
    ds = ds.apply(tf.data.experimental.ignore_errors(log_warning=True))  # failsafe so training doesn't stop
    return ds


def parse_transcription_tfexample(
    serialized_example: tf.train.Example,
) -> Tuple[
    tf.Tensor,
    tf.Tensor,
    tf.Tensor,
    tf.Tensor,
    tf.Tensor,
    tf.Tensor,
    tf.Tensor,
    tf.Tensor,
    tf.Tensor,
    tf.Tensor,
    tf.Tensor,
]:
    """
    return a tuple with the following tensors, in order:
     - file_id
     - source
     - audio_wav
     - notes_indices
     - notes_values
     - onsets_indices
     - onsets_values
     - contours_indices
     - contours_values
     - notes_onsets_shape
     - contours_shape
    NB.: notes, onsets and contours are represented as sparse matrices
    (to be reconstructed using `tf.SparseTensor(...)`). They share the
    time dimension, while contours have a frequency dimension that is
    a multiple (`ANNOTATIONS_BINS_PER_SEMITONE`) of that of
    notes/onsets.
    """
    schema = {
        "file_id": tf.io.FixedLenFeature((), tf.string),
        "source": tf.io.FixedLenFeature((), tf.string),
        "audio_wav": tf.io.FixedLenFeature((), tf.string),
        "notes_indices": tf.io.FixedLenFeature((), tf.string),
        "notes_values": tf.io.FixedLenFeature((), tf.string),
        "onsets_indices": tf.io.FixedLenFeature((), tf.string),
        "onsets_values": tf.io.FixedLenFeature((), tf.string),
        "contours_indices": tf.io.FixedLenFeature((), tf.string),
        "contours_values": tf.io.FixedLenFeature((), tf.string),
        "notes_onsets_shape": tf.io.FixedLenFeature((), tf.string),
        "contours_shape": tf.io.FixedLenFeature((), tf.string),
    }
    example = tf.io.parse_single_example(serialized_example, schema)
    return (
        example["file_id"],
        example["source"],
        example["audio_wav"],
        tf.io.parse_tensor(example["notes_indices"], out_type=tf.int64),
        tf.io.parse_tensor(example["notes_values"], out_type=tf.float32),
        tf.io.parse_tensor(example["onsets_indices"], out_type=tf.int64),
        tf.io.parse_tensor(example["onsets_values"], out_type=tf.float32),
        tf.io.parse_tensor(example["contours_indices"], out_type=tf.int64),
        tf.io.parse_tensor(example["contours_values"], out_type=tf.float32),
        tf.io.parse_tensor(example["notes_onsets_shape"], out_type=tf.int64),
        tf.io.parse_tensor(example["contours_shape"], out_type=tf.int64),
    )


def is_not_bad_shape(
    _file_id: tf.Tensor,
    _source: tf.Tensor,
    _audio_wav: tf.Tensor,
    _notes_indices: tf.Tensor,
    notes_values: tf.Tensor,
    _onsets_indices: tf.Tensor,
    _onsets_values: tf.Tensor,
    _contours_indices: tf.Tensor,
    _contours_values: tf.Tensor,
    notes_onsets_shape: tf.Tensor,
    _contours_shape: tf.Tensor,
) -> tf.Tensor:
    bad_shape = tf.logical_and(
        tf.shape(notes_values)[0] == 0,
        tf.shape(notes_onsets_shape)[0] == 2,
    )
    return tf.math.logical_not(bad_shape)


def sparse2dense(values: tf.Tensor, indices: tf.Tensor, dense_shape: tf.Tensor) -> tf.Tensor:
    if tf.rank(indices) != 2 and tf.size(indices) == 0:
        indices = tf.zeros([0, 1], dtype=indices.dtype)
    tf.assert_rank(indices, 2)
    tf.assert_rank(values, 1)
    tf.assert_rank(dense_shape, 1)
    sp = tf.SparseTensor(indices=indices, values=values, dense_shape=dense_shape)
    return tf.sparse.to_dense(sp, validate_indices=False)


def reduce_transcription_inputs(
    file_id: str,
    src: str,
    wav: Tuple[tf.Tensor, int],
    notes: tf.Tensor,
    onsets: tf.Tensor,
    contour: tf.Tensor,
) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor, Dict[str, str]]:
    """Map tf records data to a tuple
    If audio is stereo, it is mixed down to mono.
    This will error if the sample rate of the wav file is different from
    what we hard code.
    Args:
        fid : file id (string)
        src : name of dataset (string)
        wav : tensorflow wav obejct (tuple of audio and sample rate)
            The whole audio file length
        notes : matrix of note frames (n_frames x N_FREQ_BINS_NOTES)
            possibly size 0
        onsets : matrix of note onsets (n_frames x N_FREQ_BINS_NOTES)
            possibly size 0
        contours : matrix of contour frames (n_frames x N_FREQ_BINS_CONTOURS)
            possibly size 0
    """
    audio, sample_rate = wav
    tf.debugging.assert_equal(
        sample_rate,
        AUDIO_SAMPLE_RATE,
        message="audio sample rate {} is inconsistent".format(sample_rate),
    )
    return (
        tf.math.reduce_mean(audio, axis=1, keepdims=True),  # manually mixdown to mono
        onsets,
        contour,
        notes,
        {"fid": file_id, "src": src},
    )


def _infer_time_size(onsets: tf.Tensor, contour: tf.Tensor, notes: tf.Tensor) -> tf.Tensor:
    """Some of the targets might be empty, but we need to find out the
    number of time frames of one of the non-empty ones.
    Returns
        number of time frames in the targets
    """
    onset_shape = tf.shape(onsets)[0]
    contour_shape = tf.shape(contour)[0]
    note_shape = tf.shape(notes)[0]
    time_size = tf.cast(
        tf.math.maximum(
            tf.cast(tf.math.maximum(onset_shape, contour_shape), dtype=tf.int32),
            note_shape,
        ),
        dtype=tf.int32,
    )

    return time_size


def get_sample_weights(audio, onsets, contour, notes, metadata):
    """Add sample weights based on whether or not the target is empty
    If it's empty, the weight is 0, otherwise it's 1. Empty targets get filled
    with matricies of 0's
    Args:
        audio : audio signal (full length)
        notes : matrix of note frames (n_frames x N_FREQ_BINS_NOTES)
            possibly size 0
        onsets : matrix of note onsets (n_frames x N_FREQ_BINS_NOTES)
            possibly size 0
        contours : matrix of contour frames (n_frames x N_FREQ_BINS_CONTOURS)
            possibly size 0
        metadata : dictionary of metadata
    Returns:
        audio : audio signal (full length)
        notes : matrix of note frames (n_frames x N_FREQ_BINS_NOTES)
        onsets : matrix of note onsets (n_frames x N_FREQ_BINS_NOTES)
        contours : matrix of contour frames (n_frames x N_FREQ_BINS_CONTOURS)
        onset_weight : int (0 or 1)
        note_weight : int (0 or 1)
        contour_weight : int (0 or 1)
    """
    time_size = _infer_time_size(onsets, contour, notes)

    # TODO - if we dont want to worry about batches with no examples for a task
    # we can add a tiny constant here, but training will be unstable
    onsets_weight = tf.cast(tf.shape(onsets)[0] != 0, tf.float32)
    contour_weight = tf.cast(tf.shape(contour)[0] != 0, tf.float32)
    note_weight = tf.cast(tf.shape(notes)[0] != 0, tf.float32)

    onsets = tf.cond(
        tf.shape(onsets)[0] == 0,
        lambda: tf.zeros(
            tf.stack([time_size, tf.constant(N_FREQ_BINS_NOTES, dtype=tf.int32)], axis=0),
            dtype=tf.float32,
        ),
        lambda: onsets,
    )
    contour = tf.cond(
        tf.shape(contour)[0] == 0,
        lambda: tf.zeros(
            tf.stack([time_size, tf.constant(N_FREQ_BINS_CONTOURS, dtype=tf.int32)], axis=0),
            dtype=tf.float32,
        ),
        lambda: contour,
    )
    notes = tf.cond(
        tf.shape(notes)[0] == 0,
        lambda: tf.zeros(
            tf.stack([time_size, tf.constant(N_FREQ_BINS_NOTES, dtype=tf.int32)], axis=0),
            dtype=tf.float32,
        ),
        lambda: notes,
    )

    return (
        audio,
        onsets,
        contour,
        notes,
        onsets_weight,
        contour_weight,
        note_weight,
        metadata,
    )


def trim_time(data, start, duration, sr):
    """
    Slice a data file
    Args:
        data: 2D data as (n_time_samples, n_channels) array
            can be audio or a time-frequency matrix
        start: trim start time in seconds
        duration: trim duration in seconds
        sr: data sample rate
    Returns:
        sliced_data (tf.tensor): (trimmed_time, n_channels)
    """
    n_start = tf.cast(tf.math.round(sr * start), dtype=tf.int32)
    n_duration = tf.cast(tf.math.ceil(tf.cast(sr * duration, tf.float32)), dtype=tf.int32)
    begin = (n_start, 0)
    size = (n_duration, -1)
    return tf.slice(data, begin=begin, size=size)


def extract_window(audio, onsets, contour, notes, t_start):
    # needs a hop size extra of samples for good mel spectrogram alignment
    audio_trim = trim_time(
        audio,
        t_start,
        tf.cast(AUDIO_N_SAMPLES / AUDIO_SAMPLE_RATE, dtype=tf.dtypes.float32),
        AUDIO_SAMPLE_RATE,
    )
    onset_trim = trim_time(onsets, t_start, AUDIO_WINDOW_LENGTH, ANNOTATIONS_FPS)
    contour_trim = trim_time(contour, t_start, AUDIO_WINDOW_LENGTH, ANNOTATIONS_FPS)
    note_trim = trim_time(notes, t_start, AUDIO_WINDOW_LENGTH, ANNOTATIONS_FPS)
    return (audio_trim, onset_trim, contour_trim, note_trim)


def extract_random_window(audio, onsets, contour, notes, seed):
    """Trim transcription data to a fixed length of time
    starting from a random time index.
    Args:
        audio : audio signal (full length)
        notes : matrix of note frames (n_frames x N_FREQ_BINS_NOTES)
        onsets : matrix of note onsets (n_frames x N_FREQ_BINS_NOTES)
        contours : matrix of contour frames (n_frames x N_FREQ_BINS_CONTOURS)
    Returns:
        audio : audio signal (AUDIO_WINDOW_LENGTH * AUDIO_SAMPLE_RATE, 1)
        notes : matrix of note frames (AUDIO_SAMPLE_RATE * ANNOTATIONS_FPS, N_FREQ_BINS_NOTES)
        onsets : matrix of note onsets (AUDIO_SAMPLE_RATE * ANNOTATIONS_FPS, N_FREQ_BINS_NOTES)
        contours : matrix of contour frames (AUDIO_SAMPLE_RATE * ANNOTATIONS_FPS, N_FREQ_BINS_CONTOURS)
    """
    n_sec = tf.math.divide(
        tf.cast(tf.shape(audio)[0], dtype=tf.float32),
        tf.cast(AUDIO_SAMPLE_RATE, dtype=tf.float32),
    )
    t_start = tf.random.uniform(
        (),
        minval=0.0,
        maxval=n_sec - (AUDIO_N_SAMPLES / AUDIO_SAMPLE_RATE),
        dtype=tf.dtypes.float32,
        seed=seed,
    )
    return extract_window(audio, onsets, contour, notes, t_start)


def get_transcription_chunks(
    audio,
    onsets,
    contour,
    notes,
    onset_weight,
    contour_weight,
    note_weight,
    n_samples_per_track,
    seed,
):
    """Randomly sample fixed-length time chunks for transcription data
    Args:
        audio : audio signal (full length)
        notes : matrix of note frames (n_frames x N_FREQ_BINS_NOTES)
        onsets : matrix of note onsets (n_frames x N_FREQ_BINS_NOTES)
        contours : matrix of contour frames (n_frames x N_FREQ_BINS_CONTOURS)
        onset_weight : int (0 or 1)
        note_weight : int (0 or 1)
        contour_weight : int (0 or 1)
        n_samples_per_track : int - how many samples to yield per track
    Returns:
        batches of size n_samples_per_track of:
            audio : audio signal (AUDIO_WINDOW_LENGTH * AUDIO_SAMPLE_RATE, 1)
            notes : matrix of note frames (AUDIO_SAMPLE_RATE * ANNOTATIONS_FPS, N_FREQ_BINS_NOTES)
            onsets : matrix of note onsets (AUDIO_SAMPLE_RATE * ANNOTATIONS_FPS, N_FREQ_BINS_NOTES)
            contours : matrix of contour frames (AUDIO_SAMPLE_RATE * ANNOTATIONS_FPS, N_FREQ_BINS_CONTOURS)
            onset_weight : int (0 or 1)
            note_weight : int (0 or 1)
            contour_weight : int (0 or 1)
    """
    a = []
    o = []
    c = []
    n = []
    ow = []
    cw = []
    nw = []
    for i in range(n_samples_per_track):
        s0, s1, s2, s3 = extract_random_window(audio, onsets, contour, notes, i if seed else None)
        a.append(s0)
        o.append(s1)
        c.append(s2)
        n.append(s3)
        ow.append(onset_weight)
        cw.append(contour_weight)
        nw.append(note_weight)
    return tf.data.Dataset.from_tensor_slices((a, o, c, n, ow, cw, nw))


def is_not_all_silent_annotations(a, o, c, n, ow, cw, nw):
    contours_nonsilent = tf.math.reduce_mean(c) != 0
    notes_nonsilent = tf.math.reduce_mean(n) != 0
    return tf.math.logical_or(contours_nonsilent, notes_nonsilent)


def to_transcription_training_input(audio, onsets, contour, notes, onset_weight, contour_weight, note_weight):
    """convert transcription data to the format expected by the model"""
    return (
        audio,
        {
            "onset": tf.ensure_shape(onsets, (ANNOT_N_FRAMES, N_FREQ_BINS_NOTES)),
            "contour": tf.ensure_shape(contour, (ANNOT_N_FRAMES, N_FREQ_BINS_CONTOURS)),
            "note": tf.ensure_shape(notes, (ANNOT_N_FRAMES, N_FREQ_BINS_NOTES)),
        },
        {"onset": onset_weight, "contour": contour_weight, "note": note_weight},
    )
