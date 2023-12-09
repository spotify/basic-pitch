#!C:\Users\Alex\anaconda3\envs\MusicTranscriber\python
print("Content-Type: text/html\r\n\r\n")

import cgi
import os
import json
import joblib

# Enable debugging
import cgitb
cgitb.enable()

def print_response(status, data):
    json_response = {'status': status, 'data': data}
    print(json.dumps(json_response))
    exit()

# set the path of the temp file storage
TEMP_FILE_STORAGE_PATH = '../temp_file_storage/'

# Create instance of FieldStorage
form = cgi.FieldStorage()

# Get the file field
fileitem = form['audio']
selected_model_name = form['model']

cwd = os.getcwd()

# Test if the file was uploaded
if fileitem.filename:
    # strip leading path from file name to avoid directory traversal attacks
    fn = os.path.basename(fileitem.filename.replace("\\", "/" ))
    open(TEMP_FILE_STORAGE_PATH + fn, 'wb').write(fileitem.file.read())
else:
    print_response('error', 'No file was uploaded')

# Test if the model was selected
if not(selected_model_name):
    print_response('error', 'No model was selected')

# Get the file
audio_file = TEMP_FILE_STORAGE_PATH + fn

# Get the model
