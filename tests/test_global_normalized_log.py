import unittest
import numpy as np
import tensorflow as tf
from basic_pitch.layers.signal import GlobalNormalizedLog
from basic_pitch.layers import nnaudio
from basic_pitch.nn import FlattenAudioCh
from basic_pitch.constants import (
    AUDIO_SAMPLE_RATE,
    FFT_HOP,
    ANNOTATIONS_BASE_FREQUENCY,
    CONTOURS_BINS_PER_SEMITONE,
)

class TestGlobalNormalizedLog(unittest.TestCase):
    def setUp(self):
        # Create a 10-second sinusoid with increasing amplitude
        duration = 10  # seconds
        freq = 440  # Hz
        t = np.linspace(0, duration, int(duration * AUDIO_SAMPLE_RATE))
        amplitude = np.linspace(0.01, 1.0, len(t))
        self.signal = amplitude * np.sin(2 * np.pi * freq * t)
        
        # Create the processing pipeline
        self.flatten = FlattenAudioCh()
        self.cqt = nnaudio.CQT(
            sr=AUDIO_SAMPLE_RATE,
            hop_length=FFT_HOP,
            fmin=ANNOTATIONS_BASE_FREQUENCY,
            n_bins=84,  # Standard number of bins
            bins_per_octave=12 * CONTOURS_BINS_PER_SEMITONE,
        )
        self.global_norm = GlobalNormalizedLog()

    def test_global_normalized_log(self):
        # Process the signal through the pipeline
        x = self.flatten(self.signal[np.newaxis, :, np.newaxis])
        cqt_output = self.cqt(x)
        
        # Print CQT output properties
        print("\nCQT output properties:")
        print(f"CQT shape: {cqt_output.shape}")
        print(f"CQT range: {np.min(cqt_output):.3f} to {np.max(cqt_output):.3f}")
        
        # Convert to dB for verification
        power = np.square(cqt_output)
        db = 10 * np.log10(power + 1e-10)
        print(f"dB range: {np.min(db):.3f} to {np.max(db):.3f}")
        
        # Apply GlobalNormalizedLog
        normalized_output = self.global_norm(cqt_output)
        
        # Print normalized output properties
        print("\nNormalized output properties:")
        print(f"Normalized shape: {normalized_output.shape}")
        print(f"Normalized range: {np.min(normalized_output):.3f} to {np.max(normalized_output):.3f}")
        
        # Verify output range is in [0,1]
        self.assertTrue(np.all(normalized_output >= 0))
        self.assertTrue(np.all(normalized_output <= 1))
        
        # Verify output shape matches input shape
        self.assertEqual(normalized_output.shape, cqt_output.shape)
        
        # Verify dB values are properly clamped
        db_after_norm = self.global_norm.min_db + normalized_output * (self.global_norm.max_db - self.global_norm.min_db)
        self.assertTrue(np.all(db_after_norm >= self.global_norm.min_db))
        self.assertTrue(np.all(db_after_norm <= self.global_norm.max_db))

    def test_global_normalized_log_with_clipping(self):
        # Create a test signal with values that would be affected by clipping
        # We'll create a signal with some very small values (which will give very negative dB values)
        # and some very large values (which will give positive dB values)
        duration = 5  # seconds - increased from 1 to 5 to ensure enough samples for CQT
        t = np.linspace(0, duration, int(duration * AUDIO_SAMPLE_RATE))
        
        # Create a signal with varying amplitudes
        signal = np.zeros_like(t)
        signal[:len(t)//3] = 0.0001  # Very small values
        signal[len(t)//3:2*len(t)//3] = 1.0  # Normal values
        signal[2*len(t)//3:] = 10.0  # Very large values
        
        # Process through pipeline
        x = self.flatten(signal[np.newaxis, :, np.newaxis])
        cqt_output = self.cqt(x)
        
        # Print CQT output properties
        print("\nCQT output properties (with clipping test):")
        print(f"CQT shape: {cqt_output.shape}")
        print(f"CQT range: {np.min(cqt_output):.3f} to {np.max(cqt_output):.3f}")
        
        # Convert to dB for verification
        power = np.square(cqt_output)
        db = 10 * np.log10(power + 1e-10)
        print(f"dB range: {np.min(db):.3f} to {np.max(db):.3f}")
        
        # Apply GlobalNormalizedLog
        normalized_output = self.global_norm(cqt_output)
        
        # Print normalized output properties
        print("\nNormalized output properties (with clipping test):")
        print(f"Normalized shape: {normalized_output.shape}")
        print(f"Normalized range: {np.min(normalized_output):.3f} to {np.max(normalized_output):.3f}")
        
        # Verify output range is in [0,1]
        self.assertTrue(np.all(normalized_output >= 0))
        self.assertTrue(np.all(normalized_output <= 1))
        
        # Verify output shape matches input shape
        self.assertEqual(normalized_output.shape, cqt_output.shape)
        
        # Verify dB values are properly clamped
        db_after_norm = self.global_norm.min_db + normalized_output * (self.global_norm.max_db - self.global_norm.min_db)
        self.assertTrue(np.all(db_after_norm >= self.global_norm.min_db))
        self.assertTrue(np.all(db_after_norm <= self.global_norm.max_db))

    def test_global_normalized_log_direct(self):
        """Test GlobalNormalizedLog directly with values that would be affected by clipping"""
        # Create test values that would give dB values outside the [-100, 0] range
        test_values = np.array([
            [0.00001],  # Very quiet: ~-100 dB
            [0.1],      # Quiet: ~-20 dB
            [1.0],      # Normal: 0 dB
            [10.0],     # Loud: +20 dB
            [100.0],    # Very loud: +40 dB
        ])
        
        # Convert to dB for verification
        power = np.square(test_values)
        db = 10 * np.log10(power + 1e-10)
        print("\nDirect test dB values:")
        print(f"Input values: {test_values.flatten()}")
        print(f"Corresponding dB values: {db.flatten()}")
        
        # Apply GlobalNormalizedLog
        normalized_output = self.global_norm(test_values)
        normalized_output_np = normalized_output.numpy()
        
        print("\nNormalized values:")
        print(f"Output: {normalized_output_np.flatten()}")
        
        # Convert back to dB to verify clamping
        db_after_norm = self.global_norm.min_db + normalized_output_np * (self.global_norm.max_db - self.global_norm.min_db)
        print(f"dB after normalization: {db_after_norm.flatten()}")
        
        # Verify output range is in [0,1]
        self.assertTrue(np.all(normalized_output_np >= 0))
        self.assertTrue(np.all(normalized_output_np <= 1))
        
        # Verify dB values are properly clamped
        self.assertTrue(np.all(db_after_norm >= self.global_norm.min_db))
        self.assertTrue(np.all(db_after_norm <= self.global_norm.max_db))
        
        # Verify that values outside the range are clamped
        # Extract single elements properly to avoid deprecation warnings
        self.assertAlmostEqual(normalized_output_np[0, 0], 0.0, places=2)    # Very quiet (-97 dB) maps to 0.0
        self.assertAlmostEqual(normalized_output_np[1, 0], 0.42, places=2)   # -20 dB maps to ~0.42
        self.assertAlmostEqual(normalized_output_np[2, 0], 0.57, places=2)   # 0 dB maps to ~0.57
        self.assertAlmostEqual(normalized_output_np[3, 0], 0.73, places=2)   # +20 dB maps to ~0.73
        self.assertAlmostEqual(normalized_output_np[4, 0], 0.89, places=2)   # +40 dB maps to ~0.89

if __name__ == '__main__':
    unittest.main() 