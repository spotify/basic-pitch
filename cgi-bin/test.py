#!C:\Users\Alex\anaconda3\envs\MusicTranscriberClean\python
print("Content-Type: text/html\r\n\r\n")

import sys

print(sys.path)

import cgi
import os
import json

# Enable debugging
#import cgitb
#cgitb.enable()

IS_TEST = True

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



print_response('success', 'No errors')
