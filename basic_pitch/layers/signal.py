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

from typing import Any, Callable, Optional
import tensorflow as tf
from basic_pitch.layers.math import log_base_b


class Stft(tf.keras.layers.Layer):
    def __init__(
        self,
        fft_length: int = 2048,
        hop_length: Optional[int] = None,
        window_length: Optional[int] = None,
        window_fn: Callable[[int, tf.dtypes.DType], tf.Tensor] = tf.signal.hann_window,
        pad_end: bool = False,
        center: bool = True,
        pad_mode: str = "REFLECT",
        name: Optional[str] = None,
        dtype: tf.dtypes.DType = tf.float32,
    ):
        """
        A Tensorflow Keras layer that calculates an STFT.
        The input is real-valued with shape (num_batches, num_samples).
        The output is complex-valued with shape (num_batches, time, fft_length // 2 + 1)

        Args:
            hop_length: The "stride" or number of samples to iterate before the start of the next frame.
            fft_length: FFT length.
            window_length: Window length. If None, then fft_length is used.
            window_fn: A callable that takes a window length and a dtype and returns a window.
            pad_end: Whether to pad the end of signals with zeros when the provided frame length and step produces
                a frame that lies partially past its end.
            center:
                If True, the signal y is padded so that frame D[:, t] is centered at y[t * hop_length].
                If False, then D[:, t] begins at y[t * hop_length].
            pad_mode: Padding to use if center is True. One of "CONSTANT", "REFLECT", or "SYMMETRIC" (case-insensitive).
            name: Name of the layer.
            dtype: Type used in calcuation.
        """
        super().__init__(trainable=False, name=name, dtype=dtype, dynamic=False)
        self.fft_length = fft_length
        self.window_length = window_length if window_length else self.fft_length
        self.hop_length = hop_length if hop_length else self.window_length // 4
        self.window_fn = window_fn
        self.final_window_fn = window_fn
        self.pad_end = pad_end
        self.center = center
        self.pad_mode = pad_mode

    def build(self, input_shape: tf.TensorShape) -> None:
        if self.window_length < self.fft_length:
            lpad = (self.fft_length - self.window_length) // 2
            rpad = self.fft_length - self.window_length - lpad

            def padded_window(window_length: int, dtype: tf.dtypes.DType = tf.float32) -> tf.Tensor:
                # This is a trick to match librosa's way of handling window lengths < their fft_lengths
                # In that case the window is 0 padded such that the window is centered around 0s
                # In the Tensorflow case, the window is computed, multiplied against the frame and then
                # Right padded with 0's.
                return tf.pad(self.window_fn(self.window_length, dtype=dtype), [[lpad, rpad]])  # type: ignore

            self.final_window_fn = padded_window

        if self.center:
            self.spec = tf.keras.layers.Lambda(
                lambda x: tf.pad(
                    x,
                    [[0, 0] for _ in range(input_shape.rank - 1)] + [[self.fft_length // 2, self.fft_length // 2]],
                    mode=self.pad_mode,
                )
            )
        else:
            self.spec = tf.keras.layers.Lambda(lambda x: x)

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        return tf.signal.stft(
            signals=self.spec(inputs),
            frame_length=self.fft_length,
            frame_step=self.hop_length,
            fft_length=self.fft_length,
            window_fn=self.final_window_fn,
            pad_end=self.pad_end,
        )

    def get_config(self) -> Any:
        config = super().get_config().copy()
        config.update(
            {
                "fft_length": self.fft_length,
                "window_length": self.window_length,
                "hop_length": self.hop_length,
                "window_fn": self.window_fn,
                "pad_end": self.pad_end,
                "center": self.center,
                "pad_mode": self.pad_mode,
            }
        )
        return config


class Spectrogram(Stft):
    def __init__(
        self,
        power: int = 2,
        *args: Any,
        **kwargs: Any,
    ):
        """
        A Tensorflow Keras layer that calculates the magnitude spectrogram.
        The input is real-valued with shape (num_batches, num_samples).
        The output is real-valued with shape (num_batches, time, fft_length // 2 + 1)

        Args:
            power: Exponent to raise abs(stft) to.
            **kwargs: Any arguments that you'd pass to Stft
        """
        super().__init__(
            *args,
            **kwargs,
        )
        self.power = power

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        return tf.math.pow(
            tf.math.abs(super().call(inputs)),
            self.power,
        )

    def get_config(self) -> Any:
        config = super().get_config().copy()
        config.update(
            {
                "power": self.power,
            }
        )
        return config


class NormalizedLog(tf.keras.layers.Layer):
    """
    Takes an input with a shape of either (batch, x, y, z) or (batch, y, z)
    and rescales each (y, z) to dB, scaled 0 - 1.
    Only x=1 is supported.
    This layer adds 1e-10 to all values as a way to avoid NaN math.
    """

    def build(self, input_shape: tf.Tensor) -> None:
        self.squeeze_batch = lambda batch: batch
        rank = input_shape.rank
        if rank == 4:
            assert input_shape[1] == 1, "If the rank is 4, the second dimension must be length 1"
            self.squeeze_batch = lambda batch: tf.squeeze(batch, axis=1)
        else:
            assert rank == 3, f"Only ranks 3 and 4 are supported!. Received rank {rank} for {input_shape}."

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        inputs = self.squeeze_batch(inputs)  # type: ignore
        # convert magnitude to power
        power = tf.math.square(inputs)
        log_power = 10 * log_base_b(power + 1e-10, 10)

        log_power_min = tf.reshape(tf.math.reduce_min(log_power, axis=[1, 2]), [tf.shape(inputs)[0], 1, 1])
        log_power_offset = log_power - log_power_min
        log_power_offset_max = tf.reshape(
            tf.math.reduce_max(log_power_offset, axis=[1, 2]),
            [tf.shape(inputs)[0], 1, 1],
        )
        log_power_normalized = tf.math.divide_no_nan(log_power_offset, log_power_offset_max)

        return tf.reshape(log_power_normalized, tf.shape(inputs))
