#!/usr/bin/python3
print("Content-Type: text/html\r\n\r\n")

import cgi
import os
import json

# Enable debugging
import cgitb
cgitb.enable()

# set the path of the temp file storage
TEMP_FILE_STORAGE_PATH = '../temp_file_storage/'

# Create instance of FieldStorage
form = cgi.FieldStorage()

# Get the file field
fileitem = form['audio']

cwd = os.getcwd()

# Test if the file was uploaded
if fileitem.filename:
    # strip leading path from file name to avoid directory traversal attacks
    fn = os.path.basename(fileitem.filename.replace("\\", "/" ))
    open(TEMP_FILE_STORAGE_PATH + fn, 'wb').write(fileitem.file.read())

    message = 'The file "' + fn + '" was uploaded successfully'

else:
    message = 'No file was uploaded'
    exit()


json_response = {'status': 'success', 'data': message}
print(json.dumps(json_response))