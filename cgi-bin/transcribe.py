#!C:\Users\Alex\anaconda3\envs\MusicTranscriberClean\python
print("Content-Type: text/html\r\n\r\n")

import cgi
import os
import json
import sys
import io

# Save the original stdout
original_stdout = sys.stdout

# Enable debugging
#import cgitb
#cgitb.enable()

IS_TEST = False

# set paths
TEMP_FILE_STORAGE_PATH = '../temp_file_storage/'
MODEL_STORAGE_PATH = '../usable_models/'
cwd = os.getcwd()

def print_response(status, data):
    json_response = {'status': status, 'data': data}
    print(json.dumps(json_response))

audio_file = TEMP_FILE_STORAGE_PATH + 'Wii Music.mp3'

# If webpage, get file and model from form
if IS_TEST:
    audio_file = TEMP_FILE_STORAGE_PATH + 'Wii Music.mp3'
    selected_model_name = 'spotify_model'
else:
    # Create instance of FieldStorage
    form = cgi.FieldStorage()

    # Get the file field
    fileitem = form['audio']
    selected_model_name = form.getvalue('model', None)

    # Test if the file was uploaded
    if fileitem.filename:
        # strip leading path from file name to avoid directory traversal attacks
        fn = os.path.basename(fileitem.filename.replace("\\", "/" ))
        open(TEMP_FILE_STORAGE_PATH + fn, 'wb').write(fileitem.file.read())
    else:
        print_response('error', 'No file was uploaded')

    # Test if the model was selected
    if not selected_model_name or selected_model_name == '':
        print_response('error', 'No model was selected')

    # Get the file
    audio_file = TEMP_FILE_STORAGE_PATH + fn


if selected_model_name == 'our_model':
    model_path = MODEL_STORAGE_PATH + 'dec04_train_99posweight/'
elif selected_model_name == 'spotify_model':
    model_path = MODEL_STORAGE_PATH + 'icassp_2022/nmp/'

from basic_pitch.inference import predict_and_save

# check if midi file exists
midi_file = audio_file.replace('.mp3', '_basic_pitch.mid')
if os.path.exists(midi_file):
    os.remove(midi_file)

#print('current working directory: ', cwd)
# Run the inference

# Redirect stdout to capture output
captured_output = io.StringIO()
sys.stdout = captured_output

predict_and_save(model_path=model_path, output_directory=TEMP_FILE_STORAGE_PATH, audio_path_list=[audio_file], save_midi=True, sonify_midi=False, save_model_outputs=False, save_notes=False)

# Restore original stdout
sys.stdout = original_stdout
# Check if midi file exists
if os.path.exists(midi_file):
    print_response('success', midi_file)
else:
    print_response('error', 'Midi file was not created')