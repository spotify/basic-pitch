import mir_eval
import glob
import json
import numpy as np
import tensorflow as tf

from basic_pitch.constants import (
    ANNOT_N_FRAMES,
    ANNOTATIONS_N_SEMITONES,
    AUDIO_N_SAMPLES,
    N_FREQ_BINS_CONTOURS,
)

BATCH_SIZE = 3

tfkl = tf.keras.layers

# Load in the ground truth MIDI files
# glob is a pattern matching utility for files

#use maestro-v3.0.0.json to get needed files

with open('datasets/maestro/maestro-v3.0.0/maestro-v3.0.0.json', 'r') as f:
    data = json.load(f)

    midi_files = data['midi_filename']
    audio_files = data['audio_filename']



x_audio = [mir_eval.io.load_midi("datasets/maestro/" + midi_filename) for midi_filename in midi_files]
y_midi = [mir_eval.io.load_midi("datasets/maestro/" + midi_filename) for midi_filename in midi_files]




audio = np.random.uniform(size=(BATCH_SIZE, AUDIO_N_SAMPLES, 1)).astype(np.float32)
output = np.random.uniform(size=(BATCH_SIZE, ANNOT_N_FRAMES, ANNOTATIONS_N_SEMITONES * 3, 1)).astype(
    np.float32
)

import librosa

# Load audio data from file
audio_data, sr = librosa.load('path/to/audio/file.wav', sr=AUDIO_SR)

# Convert audio data to NumPy array with the same shape as `audio`
audio = np.zeros((BATCH_SIZE, AUDIO_N_SAMPLES, 1), dtype=np.float32)
audio_data = audio_data.reshape(-1, 1)
audio[:audio_data.shape[0], :audio_data.shape[1], :] = audio_data[:audio.shape[0], :audio.shape[1], :]


#gt_files = glob.glob('ground_truth/*.mid')

# Load in the model output MIDI files
#model_files = glob.glob('model_output/*.mid')
#model_midi = [mir_eval.io.load_midi(f) for f in model_files]

# Compare the two sets of MIDI files using mir_eval
#scores = mir_eval.midi.evaluate(gt_midi, model_midi)

# Print out the evaluation scores
#print(scores)
