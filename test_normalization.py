import numpy as np
import tensorflow as tf
import argparse
from basic_pitch.layers.signal import GlobalNormalizedLog, NormalizedLog
from basic_pitch.layers import nnaudio
from basic_pitch.constants import (
    AUDIO_SAMPLE_RATE, FFT_HOP, ANNOTATIONS_BASE_FREQUENCY,
    AUDIO_N_SAMPLES, AUDIO_WINDOW_LENGTH, CONTOURS_BINS_PER_SEMITONE
)
from basic_pitch.models import get_cqt, transcription_loss, weighted_transcription_loss
from basic_pitch.nn import FlattenAudioCh, FlattenFreqCh, HarmonicStacking
import matplotlib.pyplot as plt
from modify_model import MODIFIED_MODEL_PATH
from basic_pitch import ICASSP_2022_MODEL_PATH
import os

def create_increasing_sinusoid(duration_seconds=10, freq=440, sample_rate=AUDIO_SAMPLE_RATE):
    """Create a sinusoid with linearly increasing amplitude"""
    t = np.linspace(0, duration_seconds, int(duration_seconds * sample_rate))
    # Linear ramp from 0.01 to 1.0
    amplitude = np.linspace(0.01, 1.0, len(t))
    signal = amplitude * np.sin(2 * np.pi * freq * t)
    
    # Print some diagnostic information
    print("\nInput signal properties:")
    print(f"Signal shape: {signal.shape}")
    print(f"Amplitude range: {np.min(amplitude):.3f} to {np.max(amplitude):.3f}")
    print(f"Signal range: {np.min(signal):.3f} to {np.max(signal):.3f}")
    print(f"Mean amplitude: {np.mean(amplitude):.3f}")
    
    return signal

def process_chunk(chunk, model):
    """Process a single chunk through the full model"""
    # Ensure chunk has correct size
    if len(chunk) != AUDIO_N_SAMPLES:
        # Pad or truncate to correct size
        if len(chunk) < AUDIO_N_SAMPLES:
            chunk = np.pad(chunk, (0, AUDIO_N_SAMPLES - len(chunk)))
        else:
            chunk = chunk[:AUDIO_N_SAMPLES]
    
    # Add batch and channel dimensions
    chunk = tf.convert_to_tensor(chunk.reshape(1, -1, 1), dtype=tf.float32)
    
    # Get model outputs
    outputs = model(chunk)
    
    return {
        'contours': outputs['contour'].numpy(),
        'notes': outputs['note'].numpy(),
        'onsets': outputs['onset'].numpy()
    }

def plot_results(results, signal, time, freq, save_path):
    """Plot the results comparing original and modified model outputs."""
    # Concatenate contour outputs from all chunks and transpose to (freq_bins, time)
    all_contours = np.concatenate([r['contours'][0] for r in results], axis=0).T
    all_notes = np.concatenate([r['notes'][0] for r in results], axis=0).T
    all_onsets = np.concatenate([r['onsets'][0] for r in results], axis=0).T
    
    # Calculate time points for the x-axis of the model outputs
    hop_time = FFT_HOP / AUDIO_SAMPLE_RATE  # time between frames in seconds
    output_time = np.arange(all_contours.shape[1]) * hop_time
    
    # Create the figure and grid
    fig = plt.figure(figsize=(15, 12))
    gs = fig.add_gridspec(3, 2, height_ratios=[1, 2, 1])
    fig.suptitle('Comparison of Original vs Modified Model Outputs')
    
    # Plot input signal
    ax = fig.add_subplot(gs[0, 0])
    ax.plot(time, signal)
    ax.set_title('Input Signal')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Amplitude')
    
    # Print shapes for debugging
    print("\nConcatenated shapes:")
    print(f"All contours shape: {all_contours.shape}")
    print(f"All notes shape: {all_notes.shape}")
    print(f"All onsets shape: {all_onsets.shape}")
    
    # Plot contour outputs as spectrograms
    ax = fig.add_subplot(gs[1, :])
    im = ax.imshow(
        all_contours,
        aspect='auto',
        origin='lower',
        extent=[0, time[-1], 0, all_contours.shape[0]],
        cmap='magma'
    )
    ax.set_title('Model Contours')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Pitch (bins)')
    plt.colorbar(im, ax=ax, label='Contour Value')
    
    # Plot contour values at the frequency bin closest to our input frequency
    ax = fig.add_subplot(gs[2, 0])
    # Calculate the frequency bin more accurately
    # ANNOTATIONS_BASE_FREQUENCY is the base frequency (e.g. C0)
    # CONTOURS_BINS_PER_SEMITONE determines the resolution (bins per semitone)
    # First calculate semitones from base frequency, then multiply by bins per semitone
    semitones_from_base = 12 * np.log2(freq / ANNOTATIONS_BASE_FREQUENCY)
    freq_bin = int(np.round(semitones_from_base * CONTOURS_BINS_PER_SEMITONE))
    print(f"\nFrequency bin calculation:")
    print(f"Input frequency: {freq} Hz")
    print(f"Base frequency: {ANNOTATIONS_BASE_FREQUENCY} Hz")
    print(f"Semitones from base: {semitones_from_base}")
    print(f"Bins per semitone: {CONTOURS_BINS_PER_SEMITONE}")
    print(f"Calculated freq_bin: {freq_bin}")
    print(f"Total number of bins: {all_contours.shape[0]}")
    
    # Find bin with highest mean activation
    mean_activations = np.mean(all_contours, axis=1)
    max_bin = np.argmax(mean_activations)
    print(f"Bin with highest mean activation: {max_bin}")
    print(f"Mean activation at calculated bin: {mean_activations[freq_bin]}")
    print(f"Mean activation at max bin: {mean_activations[max_bin]}")
    
    # Plot both the calculated bin and the bin with highest activation
    ax.plot(output_time, all_contours[max_bin, :], label=f'Bin {max_bin} (highest activation)')
    ax.plot(output_time, all_contours[freq_bin, :], '--', label=f'Bin {freq_bin} (calculated)')
    ax.set_title('Note contour values at Input Frequency')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel("Contour Value")
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def main():
    parser = argparse.ArgumentParser(description='Test normalization methods')
    parser.add_argument('--normalization', choices=['original', 'global'], default='global',
                       help='Which normalization method to use (original or global)')
    args = parser.parse_args()
    
    # Create the test signal
    duration = 10  # seconds
    freq = 440  # Hz
    signal = create_increasing_sinusoid(duration, freq)
    
    # Register custom objects
    custom_objects = {
        'FlattenAudioCh': FlattenAudioCh,
        'FlattenFreqCh': FlattenFreqCh,
        'HarmonicStacking': HarmonicStacking,
        'CQT': nnaudio.CQT,
        'NormalizedLog': NormalizedLog,
        'GlobalNormalizedLog': GlobalNormalizedLog,
        'transcription_loss': transcription_loss,
        'weighted_transcription_loss': weighted_transcription_loss,
        '<lambda>': lambda x, y: transcription_loss(x, y, label_smoothing=0.2),
    }
    
    # Load the appropriate model
    with tf.keras.utils.custom_object_scope(custom_objects):
        if args.normalization == 'global':
            # Load original model first to create modified model
            model = tf.keras.models.load_model(str(MODIFIED_MODEL_PATH))
        else:
            model = tf.keras.models.load_model(str(ICASSP_2022_MODEL_PATH))
    
    # Process in 2-second chunks
    chunk_size = AUDIO_N_SAMPLES
    n_chunks = len(signal) // chunk_size
    
    # Store results for each chunk
    results = []
    for i in range(n_chunks):
        start = i * chunk_size
        end = start + chunk_size
        chunk = signal[start:end]
        
        chunk_results = process_chunk(chunk, model)
        results.append(chunk_results)
    
    # Create time array for plotting
    time = np.linspace(0, duration, len(signal))
    
    # Plot results
    plot_results(results, signal, time, freq, f'normalization_test_{args.normalization}.png')

if __name__ == "__main__":
    main() 