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

import logging
import os

import tensorflow as tf

from basic_pitch import visualize


class SavedModelCallback(tf.keras.callbacks.Callback):
    def __init__(self, output_path, monitor):
        self.output_savemodel_path = output_path
        self.monitor = monitor  # 'val_loss' typically
        self.best_loss_so_far = 1000000.0

    def on_epoch_end(self, epoch, logs=None):
        loss = logs.get(self.monitor)
        if loss is None:
            logging.warning("SaveModelCallback: monitored variable %s is not defined in logs, skipping" % self.monitor)
        else:
            if loss < self.best_loss_so_far:
                output_path = os.path.join(
                    self.output_savemodel_path,
                    "%d/model" % epoch,
                )
                logging.info("SaveModelCallback: saving model at iteration %d in %s" % (epoch, output_path))
                tf.saved_model.save(self.model, output_path)
                self.best_loss_so_far = loss


class VisualizeCallback(tf.keras.callbacks.Callback):
    # TODO RACHEL make this WAY faster
    def __init__(self, train_ds, validation_ds, tensorboard_dir, original_validation_ds, contours):
        super(VisualizeCallback, self).__init__()
        self.train_iter = iter(train_ds)
        self.validation_iter = iter(validation_ds)
        self.validation_ds = original_validation_ds
        self.tensorboard_dir = os.path.join(tensorboard_dir, "tensorboard_logs")
        self.file_writer = tf.summary.create_file_writer(tensorboard_dir)
        self.contours = contours

    def on_epoch_end(self, epoch, logs):
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
                self.validation_ds,
                contours=self.contours,
            )
