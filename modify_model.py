import tensorflow as tf
import numpy as np
from basic_pitch.layers.signal import GlobalNormalizedLog
from basic_pitch.models import transcription_loss, weighted_transcription_loss
from basic_pitch.nn import FlattenAudioCh, FlattenFreqCh, HarmonicStacking
from basic_pitch.layers import nnaudio, signal
from basic_pitch import ICASSP_2022_MODEL_PATH

MODIFIED_MODEL_PATH= "icassp_2022_model_modified"


def modify_model(input_model_path, output_model_path):
    # Register custom objects
    custom_objects = {
        'FlattenAudioCh': FlattenAudioCh,
        'FlattenFreqCh': FlattenFreqCh,
        'HarmonicStacking': HarmonicStacking,
        'CQT': nnaudio.CQT,
        'NormalizedLog': signal.NormalizedLog,
        'transcription_loss': transcription_loss,
        'weighted_transcription_loss': weighted_transcription_loss,
        '<lambda>': lambda x, y: transcription_loss(x, y, label_smoothing=0.2),
    }
    
    # Load the original model with custom objects
    with tf.keras.utils.custom_object_scope(custom_objects):
        original_model = tf.keras.models.load_model(input_model_path)

    # Tail model incorporating everything after CQT normalization
    # Check position of normalized log layer
    assert isinstance(original_model.layers[3], signal.NormalizedLog)
    head_model = tf.keras.Model(inputs=original_model.inputs, outputs=original_model.layers[2].output)
    tail_model = tf.keras.Model(inputs=original_model.layers[4].input, outputs=original_model.outputs)

    # Create a new model using head + global normalized log + tail
    inputs = tf.keras.Input(shape=head_model.inputs[0].shape[1:])
    x = head_model(inputs)
    x = GlobalNormalizedLog()(x)
    x = tail_model(x)
    x = {"contour": x[0], "note": x[1], "onset": x[2]}
    new_model = tf.keras.Model(inputs=inputs, outputs=x)
    new_model.save(output_model_path)
    

if __name__ == "__main__":
    input_model_path = str(ICASSP_2022_MODEL_PATH)
    modify_model(input_model_path, MODIFIED_MODEL_PATH) 