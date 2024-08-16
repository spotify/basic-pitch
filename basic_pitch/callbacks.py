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

import os

from typing import Any, Dict

import tensorflow as tf

from basic_pitch import visualize


class VisualizeCallback(tf.keras.callbacks.Callback):
    # TODO RACHEL make this WAY faster
    """
    Callback to run during training to create tensorboard visualizations per epoch.

        Attributes:
            train_ds: training dataset to use for prediction / visualization / sonification / summarization
            valid_ds: validation dataset to use for "" ""
            tensorboard_dir: directory to output "" ""
            sonify: whether to include sonifications in tensorboard
            contours: whether to plot note contours in tensorboard
    """

    def __init__(
        self,
        train_ds: tf.data.Dataset,
        validation_ds: tf.data.Dataset,
        tensorboard_dir: str,
        sonify: bool,
        contours: bool,
    ):
        super().__init__()
        self.train_iter = iter(train_ds)
        self.validation_iter = iter(validation_ds)
        self.tensorboard_dir = os.path.join(tensorboard_dir, "tensorboard_logs")
        self.file_writer = tf.summary.create_file_writer(tensorboard_dir)
        self.sonify = sonify
        self.contours = contours

    def on_epoch_end(self, epoch: int, logs: Dict[Any, Any]) -> None:
        # the first two outputs of generator needs to be the input and the targets
        train_inputs, train_targets = next(self.train_iter)[:2]
        validation_inputs, validation_targets = next(self.validation_iter)[:2]
        for stage, inputs, targets, loss in [
            ("train", train_inputs, train_targets, logs["loss"]),
            ("validation", validation_inputs, validation_targets, logs["val_loss"]),
        ]:
            outputs = self.model.predict(inputs)
            visualize.visualize_transcription(
                self.file_writer,
                stage,
                inputs,
                targets,
                outputs,
                loss,
                epoch,
                sonify=self.sonify,
                contours=self.contours,
            )
