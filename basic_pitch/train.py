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
import logging
from datetime import datetime
from typing import List

import numpy as np
import tensorflow as tf

from basic_pitch import models
from basic_pitch.callbacks import VisualizeCallback
from basic_pitch.constants import DATASET_SAMPLING_FREQUENCY
from basic_pitch.dataset import tf_example_deserialization

logging.basicConfig(level=logging.INFO)


def main(
    source: str,
    output: str,
    batch_size: int,
    shuffle_size: int,
    learning_rate: float,
    epochs: int,
    steps_per_epoch: int,
    validation_steps: int,
    size_evaluation_callback_datasets: int,
    datasets_to_use: List[str],
    dataset_sampling_frequency: np.ndarray,
    no_contours: bool,
    weighted_onset_loss: bool,
    positive_onset_weight: float,
):
    """Parse config and run training or evaluation."""
    # configuration.add_externals()
    logging.info(f"source directory: {source}")
    logging.info(f"output directory: {output}")
    logging.info(f"tensorflow version: {tf.__version__}")
    logging.info("parameters to train.main() function:")
    logging.info(f"batch_size: {batch_size}")
    logging.info(f"shuffle_size: {shuffle_size}")
    logging.info(f"learning_rate: {learning_rate}")
    logging.info(f"epochs: {epochs}")
    logging.info(f"steps_per_epoch: {steps_per_epoch}")
    logging.info(f"validation_steps: {validation_steps}")
    logging.info(f"size_evaluation_callback_datasets: {size_evaluation_callback_datasets}")
    logging.info(f"using datasets: {datasets_to_use} with frequencies {dataset_sampling_frequency}")
    logging.info(f"no_contours: {no_contours}")
    logging.info(f"weighted_onset_loss: {weighted_onset_loss}")
    logging.info(f"positive_onset_weight: {positive_onset_weight}")

    # model
    model = models.model(no_contours=no_contours)
    input_shape = list(model.input_shape)
    if input_shape[0] is None:
        input_shape[0] = batch_size
    logging.info("input_shape" + str(input_shape))

    output_shape = model.output_shape
    for k, v in output_shape.items():
        output_shape[k] = list(v)
        if v[0] is None:
            output_shape[k][0] = batch_size
    logging.info("output_shape" + str(output_shape))
    # data loaders
    train_ds, validation_ds = tf_example_deserialization.prepare_datasets(
        source,
        shuffle_size,
        batch_size,
        validation_steps,
        datasets_to_use,
        dataset_sampling_frequency,
    )

    MAX_EVAL_CBF_BATCH_SIZE = 4
    (train_visualization_ds, validation_visualization_ds,) = tf_example_deserialization.prepare_visualization_datasets(
        source,
        batch_size=min(size_evaluation_callback_datasets, MAX_EVAL_CBF_BATCH_SIZE),
        validation_steps=max(1, size_evaluation_callback_datasets // MAX_EVAL_CBF_BATCH_SIZE),
        datasets_to_use=datasets_to_use,
        dataset_sampling_frequency=dataset_sampling_frequency,
    )

    timestamp = datetime.utcnow().strftime("%Y%m%d-%H%M")
    tensorboard_log_dir = os.path.join(output, timestamp, "tensorboard")
    callbacks = [
        tf.keras.callbacks.TensorBoard(log_dir=tensorboard_log_dir, histogram_freq=1),
        tf.keras.callbacks.EarlyStopping(patience=25, verbose=2),
        tf.keras.callbacks.ReduceLROnPlateau(verbose=1, patience=10, factor=0.5),
        tf.keras.callbacks.ModelCheckpoint(filepath=os.path.join(output, timestamp, "model.best"), save_best_only=True),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(output, timestamp, "checkpoints", "model.{epoch:02d}")
        ),
        VisualizeCallback(
            train_visualization_ds,
            validation_visualization_ds,
            tensorboard_log_dir,
            validation_ds.take(validation_steps),
            not no_contours,
        ),
    ]

    if no_contours:
        loss = models.loss_no_contour(weighted=weighted_onset_loss, positive_weight=positive_onset_weight)
    else:
        loss = models.loss(weighted=weighted_onset_loss, positive_weight=positive_onset_weight)

    # train
    model.compile(
        loss=loss,
        optimizer=tf.keras.optimizers.Adam(learning_rate),
        sample_weight_mode={"contour": None, "note": None, "onset": None},
    )

    logging.info("--- Model Training specs ---")
    logging.info(f"  train_ds: {train_ds}")
    logging.info(f"  validation_ds: {validation_ds}")
    model.summary()

    model.fit(
        train_ds,
        epochs=epochs,
        callbacks=callbacks,
        steps_per_epoch=steps_per_epoch,
        validation_data=validation_ds,
        validation_steps=validation_steps,
    )


def console_entry_point():

    """From pip installed script."""
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--source", help="Path to directory containing train/validation splits.")
    parser.add_argument("--output", help="Directory to save the model in.")
    parser.add_argument("-e", "--epochs", type=int, default=500, help="Number of training epochs.")
    parser.add_argument(
        "-b",
        "--batch-size",
        type=int,
        default=16,
        help="batch size of training. Unlike Estimator API, this specifies the batch size per-GPU.",
    )
    parser.add_argument(
        "-l",
        "--learning-rate",
        type=float,
        default=0.001,
        help="ADAM optimizer learning rate",
    )
    parser.add_argument(
        "-s",
        "--steps-per-epoch",
        type=int,
        default=100,
        help="steps_per_epoch (batch) of each training loop",
    )
    parser.add_argument(
        "-v",
        "--validation-steps",
        type=int,
        default=10,
        help="validation steps (number of BATCHES) for each validation run. MUST be a positive integer",
    )
    parser.add_argument(
        "-z",
        "--training-shuffle-size",
        type=int,
        default=100,
        help="training dataset shuffle size",
    )
    parser.add_argument(
        "--size-evaluation-callback-datasets",
        type=int,
        default=4,
        help="number of elements in the dataset used by the evaluation callback function",
    )
    for dataset in DATASET_SAMPLING_FREQUENCY.keys():
        parser.add_argument(
            f"--{dataset.lower()}",
            action="store_true",
            default=False,
            help=f"Use {dataset} dataset in training",
        )
    parser.add_argument(
        "--no-contours",
        action="store_true",
        default=False,
        help="if given, trains without supervising the contour layer",
    )
    parser.add_argument(
        "--weighted-onset-loss",
        action="store_true",
        default=False,
        help="if given, trains onsets with a class-weighted loss",
    )
    parser.add_argument(
        "--positive-onset-weight",
        type=float,
        default=0.5,
        help="Positive class onset weight. Only applies when weignted onset loss is true.",
    )

    args = parser.parse_args()
    datasets_to_use = [
        dataset.lower() for dataset in DATASET_SAMPLING_FREQUENCY.keys() if getattr(args, dataset.lower().replace("-", "_"))
    ]
    dataset_sampling_frequency = [
        frequency
        for dataset, frequency in DATASET_SAMPLING_FREQUENCY.items()
        if getattr(args, dataset.lower().replace("-", "_"))
    ]
    dataset_sampling_frequency = dataset_sampling_frequency / np.sum(dataset_sampling_frequency)

    assert args.steps_per_epoch is not None
    assert args.validation_steps > 0

    main(
        args.source,
        args.output,
        args.training_shuffle_size,
        args.batch_size,
        args.learning_rate,
        args.epochs,
        args.steps_per_epoch,
        args.validation_steps,
        args.size_evaluation_callback_datasets,
        datasets_to_use,
        dataset_sampling_frequency,
        args.no_contours,
        args.weighted_onset_loss,
        args.positive_onset_weight,
    )


if __name__ == "__main__":
    console_entry_point()
