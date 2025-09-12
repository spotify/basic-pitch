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

TENSORBOARD_LOGS_SUBDIR = "tensorboard_logs"


class VisualizeCallback(tf.keras.callbacks.Callback):
    """
    Callback to run during training to create tensorboard visualizations per epoch.

    Attributes:
        train_ds: training dataset used for prediction / visualization / sonification / summarization
        validation_ds: validation dataset used for prediction / visualization / sonification / summarization
        tensorboard_dir: directory to output tensorboard logs and visualizations
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
        max_batches: int = 2,
        prefetch_batches: int = 2,
        use_tf_function: bool = True,
    ):
        super().__init__()
        self.train_ds = train_ds.take(max_batches).prefetch(prefetch_batches)
        self.validation_ds = validation_ds.take(max_batches).prefetch(prefetch_batches)
        self.tensorboard_dir = os.path.join(tensorboard_dir, TENSORBOARD_LOGS_SUBDIR)
        self.file_writer = tf.summary.create_file_writer(self.tensorboard_dir)
        self.sonify = sonify
        self.contours = contours
        self.use_tf_function = use_tf_function

    self.max_batches = max_batches

        self.train_iter = iter(self.train_ds)
        self.validation_iter = iter(self.validation_ds)

        self._predict_fn = None

    def set_model(self, model: tf.keras.Model) -> None:
        super().set_model(model)
        # if things aren't speeding up, skip tf.function and just use model.predict
        if self.use_tf_function:

            def fast_predict(inputs: tf.Tensor) -> Any:
                return model(inputs, training=False)

            self._predict_fn = tf.function(fast_predict)
        else:
            self._predict_fn = model.predict

    def _predict(self, inputs: tf.Tensor) -> Any:
        if self._predict_fn is not None:
            outputs = self._predict_fn(inputs)
            if isinstance(outputs, dict):
                outputs = {k: v.numpy() if hasattr(v, "numpy") else v for k, v in outputs.items()}
            elif isinstance(outputs, (list, tuple)):
                outputs = [v.numpy() if hasattr(v, "numpy") else v for v in outputs]
            return outputs
        else:
            return self.model.predict(inputs)

    def on_epoch_end(self, epoch: int, logs: Dict[Any, Any]) -> None:
        for stage, ds, loss_key in [
            ("train", self.train_ds, "loss"),
            ("validation", self.validation_ds, "val_loss"),
        ]:
            batch_count = 0
            for batch in ds:
                if batch_count >= self.max_batches:
                    break
                inputs, targets = batch[:2]
                outputs = self._predict(inputs)
                loss_val = logs.get(loss_key)
                try:
                    visualize.visualize_transcription(
                        self.file_writer,
                        stage,
                        inputs,
                        targets,
                        outputs,
                        float(loss_val) if loss_val is not None else 0.0,
                        epoch,
                        sonify=self.sonify,
                        contours=self.contours,
                    )
                except Exception as e:
                    print(f"Warning: Visualization failed for {stage} at epoch {epoch}: {e}")
                batch_count += 1
