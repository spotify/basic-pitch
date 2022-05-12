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
#
# This module is comprised of PyTorch layers from NNAudio and ported to TensorFlow:
# https://github.com/KinWaiCheuk/nnAudio
# The above code is released under an MIT license.

import warnings
import tensorflow as tf
import numpy as np
from typing import Any, List, Optional, Tuple, Union

import scipy.signal


def create_lowpass_filter(
    band_center: float = 0.5,
    kernel_length: int = 256,
    transition_bandwidth: float = 0.03,
    dtype: tf.dtypes.DType = tf.float32,
) -> np.ndarray:
    """
    Calculate the highest frequency we need to preserve and the lowest frequency we allow
    to pass through. Note that frequency is on a scale from 0 to 1 where 0 is 0 and 1 is
    the Nyquist frequency of the signal BEFORE downsampling.
    """

    passband_max = band_center / (1 + transition_bandwidth)
    stopband_min = band_center * (1 + transition_bandwidth)

    # We specify a list of key frequencies for which we will require
    # that the filter match a specific output gain.
    # From [0.0 to passband_max] is the frequency range we want to keep
    # untouched and [stopband_min, 1.0] is the range we want to remove
    key_frequencies = [0.0, passband_max, stopband_min, 1.0]

    # We specify a list of output gains to correspond to the key
    # frequencies listed above.
    # The first two gains are 1.0 because they correspond to the first
    # two key frequencies. the second two are 0.0 because they
    # correspond to the stopband frequencies
    gain_at_key_frequencies = [1.0, 1.0, 0.0, 0.0]

    # This command produces the filter kernel coefficients
    filter_kernel = scipy.signal.firwin2(kernel_length, key_frequencies, gain_at_key_frequencies)

    return tf.constant(filter_kernel, dtype=dtype)


def next_power_of_2(A: int) -> int:
    """A helper function to calculate the next nearest number to the power of 2."""
    return int(np.ceil(np.log2(A)))


def early_downsample(
    sr: Union[float, int],
    hop_length: int,
    n_octaves: int,
    nyquist_hz: float,
    filter_cutoff_hz: float,
) -> Tuple[Union[float, int], int, int]:
    """Return new sampling rate and hop length after early downsampling"""
    downsample_count = early_downsample_count(nyquist_hz, filter_cutoff_hz, hop_length, n_octaves)
    downsample_factor = 2 ** (downsample_count)

    hop_length //= downsample_factor  # Getting new hop_length
    new_sr = sr / float(downsample_factor)  # Getting new sampling rate

    return new_sr, hop_length, downsample_factor


# The following two downsampling count functions are obtained from librosa CQT
# They are used to determine the number of pre resamplings if the starting and ending frequency
# are both in low frequency regions.
def early_downsample_count(nyquist_hz: float, filter_cutoff_hz: float, hop_length: int, n_octaves: int) -> int:
    """Compute the number of early downsampling operations"""

    downsample_count1 = max(0, int(np.ceil(np.log2(0.85 * nyquist_hz / filter_cutoff_hz)) - 1) - 1)
    num_twos = next_power_of_2(hop_length)
    downsample_count2 = max(0, num_twos - n_octaves + 1)

    return min(downsample_count1, downsample_count2)


def get_early_downsample_params(
    sr: Union[float, int], hop_length: int, fmax_t: float, Q: float, n_octaves: int, dtype: tf.dtypes.DType
) -> Tuple[Union[float, int], int, float, np.array, bool]:
    """Compute downsampling parameters used for early downsampling"""

    window_bandwidth = 1.5  # for hann window
    filter_cutoff = fmax_t * (1 + 0.5 * window_bandwidth / Q)
    sr, hop_length, downsample_factor = early_downsample(sr, hop_length, n_octaves, sr // 2, filter_cutoff)
    if downsample_factor != 1:
        earlydownsample = True
        early_downsample_filter = create_lowpass_filter(
            band_center=1 / downsample_factor,
            kernel_length=256,
            transition_bandwidth=0.03,
            dtype=dtype,
        )
    else:
        early_downsample_filter = None
        earlydownsample = False

    return sr, hop_length, downsample_factor, early_downsample_filter, earlydownsample


def get_window_dispatch(window: Union[str, Tuple[str, float]], N: int, fftbins: bool = True) -> np.array:
    if isinstance(window, str):
        return scipy.signal.get_window(window, N, fftbins=fftbins)
    elif isinstance(window, tuple):
        if window[0] == "gaussian":
            assert window[1] >= 0
            sigma = np.floor(-N / 2 / np.sqrt(-2 * np.log(10 ** (-window[1] / 20))))
            return scipy.signal.get_window(("gaussian", sigma), N, fftbins=fftbins)
        else:
            Warning("Tuple windows may have undesired behaviour regarding Q factor")
    elif isinstance(window, float):
        Warning("You are using Kaiser window with beta factor " + str(window) + ". Correct behaviour not checked.")
    else:
        raise Exception("The function get_window from scipy only supports strings, tuples and floats.")


def create_cqt_kernels(
    Q: float,
    fs: float,
    fmin: float,
    n_bins: int = 84,
    bins_per_octave: int = 12,
    norm: int = 1,
    window: str = "hann",
    fmax: Optional[float] = None,
    topbin_check: bool = True,
) -> Tuple[np.array, int, np.array, np.array]:
    """
    Automatically create CQT kernels in time domain
    """

    fftLen = 2 ** next_power_of_2(np.ceil(Q * fs / fmin))

    if (fmax is not None) and (n_bins is None):
        n_bins = np.ceil(bins_per_octave * np.log2(fmax / fmin))  # Calculate the number of bins
        freqs = fmin * 2.0 ** (np.r_[0:n_bins] / np.float(bins_per_octave))

    elif (fmax is None) and (n_bins is not None):
        freqs = fmin * 2.0 ** (np.r_[0:n_bins] / np.float(bins_per_octave))

    else:
        warnings.warn("If fmax is given, n_bins will be ignored", SyntaxWarning)
        n_bins = np.ceil(bins_per_octave * np.log2(fmax / fmin))  # Calculate the number of bins
        freqs = fmin * 2.0 ** (np.r_[0:n_bins] / np.float(bins_per_octave))

    if np.max(freqs) > fs / 2 and topbin_check is True:
        raise ValueError(
            "The top bin {}Hz has exceeded the Nyquist frequency, please reduce the n_bins".format(np.max(freqs))
        )

    tempKernel = np.zeros((int(n_bins), int(fftLen)), dtype=np.complex64)

    lengths = np.ceil(Q * fs / freqs)
    for k in range(0, int(n_bins)):
        freq = freqs[k]
        _l = np.ceil(Q * fs / freq)

        # Centering the kernels, pad more zeros on RHS
        start = int(np.ceil(fftLen / 2.0 - _l / 2.0)) - int(_l % 2)

        sig = (
            get_window_dispatch(window, int(_l), fftbins=True)
            * np.exp(np.r_[-_l // 2 : _l // 2] * 1j * 2 * np.pi * freq / fs)
            / _l
        )

        if norm:  # Normalizing the filter # Trying to normalize like librosa
            tempKernel[k, start : start + int(_l)] = sig / np.linalg.norm(sig, norm)
        else:
            tempKernel[k, start : start + int(_l)] = sig

    return tempKernel, fftLen, lengths, freqs


def get_cqt_complex(
    x: tf.Tensor,
    cqt_kernels_real: tf.Tensor,
    cqt_kernels_imag: tf.Tensor,
    hop_length: int,
    padding: tf.keras.layers.Layer,
) -> tf.Tensor:
    """Multiplying the STFT result with the cqt_kernel, check out the 1992 CQT paper [1]
    for how to multiple the STFT result with the CQT kernel
    [2] Brown, Judith C.C. and Miller Puckette. “An efficient algorithm for the calculation of
    a constant Q transform.” (1992)."""

    try:
        x = padding(x)  # When center is True, we need padding at the beginning and ending
    except Exception:
        warnings.warn(
            f"\ninput size = {x.shape}\tkernel size = {cqt_kernels_real.shape[-1]}\n"
            "padding with reflection mode might not be the best choice, try using constant padding",
            UserWarning,
        )
        x = tf.pad(x, (cqt_kernels_real.shape[-1] // 2, cqt_kernels_real.shape[-1] // 2))
    CQT_real = tf.transpose(
        tf.nn.conv1d(
            tf.transpose(x, [0, 2, 1]),
            tf.transpose(cqt_kernels_real, [2, 1, 0]),
            padding="VALID",
            stride=hop_length,
        ),
        [0, 2, 1],
    )
    CQT_imag = -tf.transpose(
        tf.nn.conv1d(
            tf.transpose(x, [0, 2, 1]),
            tf.transpose(cqt_kernels_imag, [2, 1, 0]),
            padding="VALID",
            stride=hop_length,
        ),
        [0, 2, 1],
    )

    return tf.stack((CQT_real, CQT_imag), axis=-1)


def downsampling_by_n(x: tf.Tensor, filter_kernel: tf.Tensor, n: float, match_torch_exactly: bool = True) -> tf.Tensor:
    """
    Downsample the given tensor using the given filter kernel.
    The input tensor is expected to have shape `(n_batches, channels, width)`,
    and the filter kernel is expected to have shape `(num_output_channels,)` (i.e.: 1D)

    If match_torch_exactly is passed, we manually pad the input rather than having TensorFlow do so with "SAME".
    The result is subtly different than Torch's output, but it is compatible with TensorFlow Lite (as of v2.4.1).
    """

    if match_torch_exactly:
        paddings = [
            [0, 0],
            [0, 0],
            [(filter_kernel.shape[-1] - 1) // 2, (filter_kernel.shape[-1] - 1) // 2],
        ]
        padded = tf.pad(x, paddings)

        # Store this tensor in the shape `(n_batches, width, channels)`
        padded_nwc = tf.transpose(padded, [0, 2, 1])
        result_nwc = tf.nn.conv1d(padded_nwc, filter_kernel[:, None, None], padding="VALID", stride=n)
    else:
        x_nwc = tf.transpose(x, [0, 2, 1])
        result_nwc = tf.nn.conv1d(x_nwc, filter_kernel[:, None, None], padding="SAME", stride=n)
    result_ncw = tf.transpose(result_nwc, [0, 2, 1])
    return result_ncw


class ReflectionPad1D(tf.keras.layers.Layer):
    """
    Replica of Torch's nn.ReflectionPad1D in TF.
    """

    def __init__(self, padding: Union[int, Tuple[int]] = 1, **kwargs: Any):
        self.padding = padding
        self.input_spec = [tf.keras.layers.InputSpec(ndim=3)]
        super(ReflectionPad1D, self).__init__(**kwargs)

    def compute_output_shape(self, s: List[int]) -> Tuple[int, int, int]:
        return (s[0], s[1], s[2] + 2 * self.padding if isinstance(self.padding, int) else self.padding[0])

    def call(self, x: tf.Tensor) -> tf.Tensor:
        return tf.pad(x, [[0, 0], [0, 0], [self.padding, self.padding]], "REFLECT")


class ConstantPad1D(tf.keras.layers.Layer):
    """
    Replica of Torch's nn.ConstantPad1D in TF.
    """

    def __init__(self, padding: Union[int, Tuple[int]] = 1, value: int = 0, **kwargs: Any):
        self.padding = padding
        self.value = value
        self.input_spec = [tf.keras.layers.InputSpec(ndim=3)]
        super(ConstantPad1D, self).__init__(**kwargs)

    def compute_output_shape(self, s: List[int]) -> Tuple[int, int, int]:
        return (s[0], s[1], s[2] + 2 * self.padding if isinstance(self.padding, int) else self.padding[0])

    def call(self, x: tf.Tensor) -> tf.Tensor:
        return tf.pad(x, [[0, 0], [0, 0], [self.padding, self.padding]], "CONSTANT", self.value)


def pad_center(data: np.ndarray, size: int, axis: int = -1, **kwargs: Any) -> np.ndarray:
    """Wrapper for np.pad to automatically center an array prior to padding.
    This is analogous to `str.center()`

    Examples
    --------
    >>> # Generate a vector
    >>> data = np.ones(5)
    >>> librosa.util.pad_center(data, 10, mode='constant')
    array([ 0.,  0.,  1.,  1.,  1.,  1.,  1.,  0.,  0.,  0.])
    >>> # Pad a matrix along its first dimension
    >>> data = np.ones((3, 5))
    >>> librosa.util.pad_center(data, 7, axis=0)
    array([[ 0.,  0.,  0.,  0.,  0.],
           [ 0.,  0.,  0.,  0.,  0.],
           [ 1.,  1.,  1.,  1.,  1.],
           [ 1.,  1.,  1.,  1.,  1.],
           [ 1.,  1.,  1.,  1.,  1.],
           [ 0.,  0.,  0.,  0.,  0.],
           [ 0.,  0.,  0.,  0.,  0.]])
    >>> # Or its second dimension
    >>> librosa.util.pad_center(data, 7, axis=1)
    array([[ 0.,  1.,  1.,  1.,  1.,  1.,  0.],
           [ 0.,  1.,  1.,  1.,  1.,  1.,  0.],
           [ 0.,  1.,  1.,  1.,  1.,  1.,  0.]])

    Parameters
    ----------
    data : np.ndarray
        Vector to be padded and centered
    size : int >= len(data) [scalar]
        Length to pad `data`
    axis : int
        Axis along which to pad and center the data
    kwargs : additional keyword arguments
      arguments passed to `np.pad()`

    Returns
    -------
    data_padded : np.ndarray
        `data` centered and padded to length `size` along the
        specified axis

    Raises
    ------
    ValueError
        If `size < data.shape[axis]`

    See Also
    --------
    numpy.pad
    """

    kwargs.setdefault("mode", "constant")

    n = data.shape[axis]

    lpad = int((size - n) // 2)

    lengths = [(0, 0)] * data.ndim
    lengths[axis] = (lpad, int(size - n - lpad))

    if lpad < 0:
        raise ValueError(("Target size ({:d}) must be at least input size ({:d})").format(size, n))

    return np.pad(data, lengths, **kwargs)


class CQT2010v2(tf.keras.layers.Layer):
    """This layer calculates the CQT of the input signal.
    Input signal should be in either of the following shapes.
    1. (len_audio)
    2. (num_audio, len_audio)
    3. (num_audio, 1, len_audio)
    The correct shape will be inferred autommatically if the input follows these 3 shapes.
    Most of the arguments follow the convention from librosa.

    This layer uses about 1MB of memory per second of input audio with its default arguments.

    This alogrithm uses the resampling method proposed in [1].
    Instead of convoluting the STFT results with a gigantic CQT kernel covering the full frequency
    spectrum, we make a small CQT kernel covering only the top octave. Then we keep downsampling the
    input audio by a factor of 2 to convoluting it with the small CQT kernel.
    Everytime the input audio is downsampled, the CQT relative to the downsampled input is equivalent
    to the next lower octave.
    The kernel creation process is still same as the 1992 algorithm. Therefore, we can reuse the
    code from the 1992 alogrithm [2]
    [1] Schörkhuber, Christian. “CONSTANT-Q TRANSFORM TOOLBOX FOR MUSIC PROCESSING.” (2010).
    [2] Brown, Judith C.C. and Miller Puckette. “An efficient algorithm for the calculation of a
    constant Q transform.” (1992).
    Early downsampling factor is to downsample the input audio to reduce the CQT kernel size.
    The result with and without early downsampling are more or less the same except in the very low
    frequency region where freq < 40Hz.
    Parameters
    ----------
    sr : int
        The sampling rate for the input audio. It is used to calucate the correct ``fmin`` and ``fmax``.
        Setting the correct sampling rate is very important for calculating the correct frequency.
    hop_length : int
        The hop (or stride) size. Default value is 512.
    fmin : float
        The frequency for the lowest CQT bin. Default is 32.70Hz, which coresponds to the note C0.
    fmax : float
        The frequency for the highest CQT bin. Default is ``None``, therefore the higest CQT bin is
        inferred from the ``n_bins`` and ``bins_per_octave``.  If ``fmax`` is not ``None``, then the
        argument ``n_bins`` will be ignored and ``n_bins`` will be calculated automatically.
        Default is ``None``
    n_bins : int
        The total numbers of CQT bins. Default is 84. Will be ignored if ``fmax`` is not ``None``.
    bins_per_octave : int
        Number of bins per octave. Default is 12.
    norm : bool
        Normalization for the CQT result.
    basis_norm : int
        Normalization for the CQT kernels. ``1`` means L1 normalization, and ``2`` means L2 normalization.
        Default is ``1``, which is same as the normalization used in librosa.
    window : str
        The windowing function for CQT. It uses ``scipy.signal.get_window``, please refer to
        scipy documentation for possible windowing functions. The default value is 'hann'
    pad_mode : str
        The padding method. Default value is 'reflect'.
    trainable : bool
        Determine if the CQT kernels are trainable or not. If ``True``, the gradients for CQT kernels
        will also be caluclated and the CQT kernels will be updated during model training.
        Default value is ``False``
    output_format : str
        Determine the return type.
        'Magnitude' will return the magnitude of the STFT result, shape = ``(num_samples, freq_bins, time_steps)``;
        'Complex' will return the STFT result in complex number, shape = ``(num_samples, freq_bins, time_steps, 2)``;
        'Phase' will return the phase of the STFT reuslt, shape = ``(num_samples, freq_bins,time_steps, 2)``.
        The complex number is stored as ``(real, imag)`` in the last axis. Default value is 'Magnitude'.
    verbose : bool
        If ``True``, it shows layer information. If ``False``, it suppresses all prints.
    device : str
        Choose which device to initialize this layer. Default value is 'cpu'.
    Returns
    -------
    spectrogram : tf.Tensor

    It returns a tensor of spectrograms.
    shape = ``(num_samples, freq_bins,time_steps)`` if ``output_format='Magnitude'``;
    shape = ``(num_samples, freq_bins,time_steps, 2)`` if ``output_format='Complex' or 'Phase'``;
    Examples
    --------
    >>> spec_layer = Spectrogram.CQT2010v2()
    >>> specs = spec_layer(x)
    """

    def __init__(
        self,
        sr: int = 22050,
        hop_length: int = 512,
        fmin: float = 32.70,
        fmax: Optional[float] = None,
        n_bins: int = 84,
        filter_scale: int = 1,
        bins_per_octave: int = 12,
        norm: bool = True,
        basis_norm: int = 1,
        window: str = "hann",
        pad_mode: str = "reflect",
        earlydownsample: bool = True,
        trainable: bool = False,
        output_format: str = "Magnitude",
        match_torch_exactly: bool = True,
    ):
        super().__init__()

        self.sample_rate: Union[float, int] = sr
        self.hop_length = hop_length
        self.fmin = fmin
        self.fmax = fmax
        self.n_bins = n_bins
        self.filter_scale = filter_scale
        self.bins_per_octave = bins_per_octave
        self.norm = norm
        self.basis_norm = basis_norm
        self.window = window
        self.pad_mode = pad_mode
        self.earlydownsample = earlydownsample
        self.trainable = trainable
        self.output_format = output_format
        self.match_torch_exactly = match_torch_exactly
        self.normalization_type = "librosa"

    def get_config(self) -> Any:
        config = super().get_config().copy()
        config.update(
            {
                "sample_rate": self.sample_rate,
                "hop_length": self.hop_length,
                "fmin": self.fmin,
                "fmax": self.fmax,
                "n_bins": self.n_bins,
                "filter_scale": self.filter_scale,
                "bins_per_octave": self.bins_per_octave,
                "norm": self.norm,
                "basis_norm": self.basis_norm,
                "window": self.window,
                "pad_mode": self.pad_mode,
                "output_format": self.output_format,
                "earlydownsample": self.earlydownsample,
                "trainable": self.trainable,
                "match_torch_exactly": self.match_torch_exactly,
            }
        )
        return config

    def build(self, input_shape: tf.TensorShape) -> None:
        # This will be used to calculate filter_cutoff and creating CQT kernels
        Q = float(self.filter_scale) / (2 ** (1 / self.bins_per_octave) - 1)

        self.lowpass_filter = create_lowpass_filter(band_center=0.5, kernel_length=256, transition_bandwidth=0.001)

        # Calculate num of filter requires for the kernel
        # n_octaves determines how many resampling requires for the CQT
        n_filters = min(self.bins_per_octave, self.n_bins)
        self.n_octaves = int(np.ceil(float(self.n_bins) / self.bins_per_octave))

        # Calculate the lowest frequency bin for the top octave kernel
        self.fmin_t = self.fmin * 2 ** (self.n_octaves - 1)
        remainder = self.n_bins % self.bins_per_octave

        if remainder == 0:
            # Calculate the top bin frequency
            fmax_t = self.fmin_t * 2 ** ((self.bins_per_octave - 1) / self.bins_per_octave)
        else:
            # Calculate the top bin frequency
            fmax_t = self.fmin_t * 2 ** ((remainder - 1) / self.bins_per_octave)

        self.fmin_t = fmax_t / 2 ** (1 - 1 / self.bins_per_octave)  # Adjusting the top minium bins
        if fmax_t > self.sample_rate / 2:
            raise ValueError(
                "The top bin {}Hz has exceeded the Nyquist frequency, please reduce the n_bins".format(fmax_t)
            )

        if self.earlydownsample is True:  # Do early downsampling if this argument is True
            (
                self.sample_rate,
                self.hop_length,
                self.downsample_factor,
                early_downsample_filter,
                self.earlydownsample,
            ) = get_early_downsample_params(self.sample_rate, self.hop_length, fmax_t, Q, self.n_octaves, self.dtype)

            self.early_downsample_filter = early_downsample_filter
        else:
            self.downsample_factor = 1.0

        # Preparing CQT kernels
        basis, self.n_fft, _, _ = create_cqt_kernels(
            Q,
            self.sample_rate,
            self.fmin_t,
            n_filters,
            self.bins_per_octave,
            norm=self.basis_norm,
            topbin_check=False,
        )

        # For the normalization in the end
        # The freqs returned by create_cqt_kernels cannot be used
        # Since that returns only the top octave bins
        # We need the information for all freq bin
        freqs = self.fmin * 2.0 ** (np.r_[0 : self.n_bins] / np.float(self.bins_per_octave))
        self.frequencies = freqs

        self.lengths = np.ceil(Q * self.sample_rate / freqs)

        self.basis = basis
        # NOTE(psobot): this is where the implementation here starts to differ from CQT2010.

        # These cqt_kernel is already in the frequency domain
        self.cqt_kernels_real = tf.expand_dims(basis.real.astype(self.dtype), 1)
        self.cqt_kernels_imag = tf.expand_dims(basis.imag.astype(self.dtype), 1)

        if self.trainable:
            self.cqt_kernels_real = tf.Variable(initial_value=self.cqt_kernels_real, trainable=True)
            self.cqt_kernels_imag = tf.Variable(initial_value=self.cqt_kernels_imag, trainable=True)

        # If center==True, the STFT window will be put in the middle, and paddings at the beginning
        # and ending are required.
        if self.pad_mode == "constant":
            self.padding = ConstantPad1D(self.n_fft // 2, 0)
        elif self.pad_mode == "reflect":
            self.padding = ReflectionPad1D(self.n_fft // 2)

        rank = len(input_shape)
        if rank == 2:
            self.reshape_input = lambda x: x[:, None, :]
        elif rank == 1:
            self.reshape_input = lambda x: x[None, None, :]
        elif rank == 3:
            self.reshape_input = lambda x: x
        else:
            raise ValueError(f"Input shape must be rank <= 3, found shape {input_shape}")

    def call(self, x: tf.Tensor) -> tf.Tensor:
        x = self.reshape_input(x)  # type: ignore

        if self.earlydownsample is True:
            x = downsampling_by_n(x, self.early_downsample_filter, self.downsample_factor, self.match_torch_exactly)

        hop = self.hop_length

        # Getting the top octave CQT
        CQT = get_cqt_complex(x, self.cqt_kernels_real, self.cqt_kernels_imag, hop, self.padding)

        x_down = x  # Preparing a new variable for downsampling

        for _ in range(self.n_octaves - 1):
            hop = hop // 2
            x_down = downsampling_by_n(x_down, self.lowpass_filter, 2, self.match_torch_exactly)
            CQT1 = get_cqt_complex(x_down, self.cqt_kernels_real, self.cqt_kernels_imag, hop, self.padding)
            CQT = tf.concat((CQT1, CQT), axis=1)

        CQT = CQT[:, -self.n_bins :, :]  # Removing unwanted bottom bins

        # Normalizing the output with the downsampling factor, 2**(self.n_octaves-1) is make it
        # same mag as 1992
        CQT = CQT * self.downsample_factor

        # Normalize again to get same result as librosa
        if self.normalization_type == "librosa":
            CQT *= tf.math.sqrt(tf.cast(self.lengths.reshape((-1, 1, 1)), self.dtype))
        elif self.normalization_type == "convolutional":
            pass
        elif self.normalization_type == "wrap":
            CQT *= 2
        else:
            raise ValueError("The normalization_type %r is not part of our current options." % self.normalization_type)

        # Transpose the output to match the output of the other spectrogram layers.
        if self.output_format.lower() == "magnitude":
            # Getting CQT Amplitude
            return tf.transpose(tf.math.sqrt(tf.math.reduce_sum(tf.math.pow(CQT, 2), axis=-1)), [0, 2, 1])

        elif self.output_format.lower() == "complex":
            return CQT

        elif self.output_format.lower() == "phase":
            phase_real = tf.math.cos(tf.math.atan2(CQT[:, :, :, 1], CQT[:, :, :, 0]))
            phase_imag = tf.math.sin(tf.math.atan2(CQT[:, :, :, 1], CQT[:, :, :, 0]))
            return tf.stack((phase_real, phase_imag), axis=-1)


CQT = CQT2010v2
