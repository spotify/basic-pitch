#!/usr/bin/env python
# encoding: utf-8

import numpy as np
import tensorflow as tf
from basic_pitch.layers import nnaudio
from basic_pitch.nn import FlattenAudioCh
from basic_pitch.constants import (
    AUDIO_SAMPLE_RATE,
    FFT_HOP,
    ANNOTATIONS_BASE_FREQUENCY,
    CONTOURS_BINS_PER_SEMITONE,
    AUDIO_N_SAMPLES,
)
from basic_pitch.layers.math import log_base_b

def create_sinusoid(freq=440, duration=None, amplitude=1.0):
    """Create a sinusoid with given frequency, duration, and amplitude."""
    if duration is None:
        # Use the exact number of samples expected by the model
        n_samples = AUDIO_N_SAMPLES
    else:
        n_samples = int(duration * AUDIO_SAMPLE_RATE)
    t = np.linspace(0, n_samples/AUDIO_SAMPLE_RATE, n_samples)
    return amplitude * np.sin(2 * np.pi * freq * t)

def process_audio(audio):
    """Process audio through CQT and return dB values."""
    # Reshape audio to match expected format (batch, time, channels)
    audio = tf.convert_to_tensor(audio, dtype=tf.float32)
    audio = tf.reshape(audio, (1, -1, 1))
    
    # Create the processing pipeline
    flatten = FlattenAudioCh()
    cqt = nnaudio.CQT(
        sr=AUDIO_SAMPLE_RATE,
        hop_length=FFT_HOP,
        fmin=ANNOTATIONS_BASE_FREQUENCY,
        n_bins=84,  # Standard number of bins
        bins_per_octave=12 * CONTOURS_BINS_PER_SEMITONE,
        pad_mode="constant",  # Use constant padding instead of reflect
    )
    
    # Process the audio
    x = flatten(audio)
    x = cqt(x)
    
    # Convert to power and then to dB
    power = tf.math.square(x)
    log_power = 10 * log_base_b(power + 1e-10, 10)
    
    return log_power

def analyze_frequency_range(min_freq, max_freq, n_freqs, amplitude):
    """Analyze CQT response across a range of frequencies."""
    freqs = np.geomspace(min_freq, max_freq, n_freqs)  # Use geometric spacing for musical frequencies
    min_peak_db = float('inf')  # Minimum of the peak responses
    max_db = float('-inf')
    mean_db = 0
    max_freq_found = 0
    min_peak_freq_found = 0
    
    print(f"\nAnalyzing frequencies from {min_freq:.1f}Hz to {max_freq:.1f}Hz at amplitude {amplitude}:")
    for freq in freqs:
        signal = create_sinusoid(freq=freq, amplitude=amplitude)
        db_values = process_audio(signal)
        
        # Find the peak response for this frequency
        curr_peak = float(tf.reduce_max(db_values))
        curr_mean = float(tf.reduce_mean(db_values))
        
        if curr_peak > max_db:
            max_db = curr_peak
            max_freq_found = freq
        if curr_peak < min_peak_db:
            min_peak_db = curr_peak
            min_peak_freq_found = freq
        mean_db += curr_mean
    
    mean_db /= len(freqs)
    
    print(f"  Minimum peak dB: {min_peak_db:.2f} (at {min_peak_freq_found:.1f}Hz)")
    print(f"  Maximum dB: {max_db:.2f} (at {max_freq_found:.1f}Hz)")
    print(f"  Mean dB: {mean_db:.2f}")
    
    return min_peak_db, max_db, mean_db, min_peak_freq_found, max_freq_found

def main():
    # Test frequencies from just below the base frequency to just below Nyquist
    min_freq = ANNOTATIONS_BASE_FREQUENCY / 2  # Test below the base frequency
    max_freq = AUDIO_SAMPLE_RATE / 2.5  # Stay comfortably below Nyquist
    n_freqs = 50  # Number of frequencies to test
    
    # Test both low and high amplitude signals
    print("\n=== Testing frequency response across the spectrum ===")
    low_results = analyze_frequency_range(min_freq, max_freq, n_freqs, amplitude=0.01)
    high_results = analyze_frequency_range(min_freq, max_freq, n_freqs, amplitude=1.0)
    
    # Print the overall extremes
    print("\n=== Overall Extremes ===")
    print(f"Minimum peak dB: {min(low_results[0], high_results[0]):.2f}")
    print(f"Absolute maximum dB: {max(low_results[1], high_results[1]):.2f}")
    
    # Calculate the maximum dB difference between amplitudes
    db_difference = high_results[1] - low_results[1]
    print(f"\nMaximum dB difference between amplitudes: {db_difference:.2f}")
    print(f"Expected dB difference: {20 * np.log10(1.0/0.01):.2f}")

if __name__ == "__main__":
    main() 