import cgi
import os
from basic_pitch.inference import predict_and_save
import uvicorn
from fastapi import FastAPI, Form, File, UploadFile
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from typing_extensions import Annotated

app = FastAPI()

origins = [
    "*",
    "http://www.alexjfisher.com/*",
    "http://www.alexjfisher.com",
    "http://localhost:*",
    "http://alexjfisher.com/*",
    "http://alexjfisher.com",
    "http://www.alexjfisher.com/projects/audio_to_midi_transcriber/index.html"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

cwd = os.getcwd()

# set paths
TEMP_FILE_STORAGE_PATH = cwd + '/src/temp_file_storage/'
MODEL_STORAGE_PATH = cwd + '/src/usable_models/'

app.mount("/static", StaticFiles(directory=TEMP_FILE_STORAGE_PATH), name="static")

@app.get('/')
def hello_world():
    return {'status': 'success', 'data': 'Hello, Dockerized Web App!'}

@app.post('/upload_file')
async def upload(model: Annotated[str, Form(...)], file: UploadFile = File(...)):
    try:
        if file.filename:
            # strip leading path from file name to avoid directory traversal attacks
            fn = os.path.basename(file.filename.replace("\\", "/" ))
            open(TEMP_FILE_STORAGE_PATH + fn, 'wb').write(file.file.read())
        else:
            return {'status': 'error', 'data': 'No file was uploaded'}
    except Exception as e:
        return {'status': 'error', 'data': 'There was an error uploading the file: ' + str(e)}
    finally:
        file.file.close()

    return {'status': 'success', 'data': {'message': f'Successfully uploaded {file.filename}', 'model': model}}

@app.post('/transcribe')
async def transcribe(model: Annotated[str, Form(...)], file: UploadFile = File(...)):
    try:
        if file.filename:
            # strip leading path from file name to avoid directory traversal attacks
            fn = os.path.basename(file.filename.replace("\\", "/" ))
            open(TEMP_FILE_STORAGE_PATH + fn, 'wb').write(file.file.read())
        else:
            return {'status': 'error', 'data': 'No file was uploaded'}
    except Exception as e:
        return {'status': 'error', 'data': 'There was an error uploading the file: ' + str(e)}
    finally:
        file.file.close()

    # Get the file
    audio_file = TEMP_FILE_STORAGE_PATH + fn

    if model == 'our_model':
        model_path = MODEL_STORAGE_PATH + 'dec04_train_99posweight/'
    elif model == 'spotify_model':
        model_path = MODEL_STORAGE_PATH + 'icassp_2022/nmp/'
    else:
        return {'status': 'error', 'data': 'Invalid model'}

    # check if midi file exists
    midi_file = audio_file.replace('.mp3', '_basic_pitch.mid')
    if os.path.exists(midi_file):
        os.remove(midi_file)

    #print('current working directory: ', cwd)
    # Run the inference

    predict_and_save(model_path=model_path, output_directory=TEMP_FILE_STORAGE_PATH, audio_path_list=[audio_file], save_midi=True, sonify_midi=False, save_model_outputs=False, save_notes=False)

    # Check if midi file exists
    if os.path.exists(midi_file):
        # get the file basename
        midi_file = os.path.basename(midi_file)
        return {'status': 'success', 'data': midi_file}
    else:
        return {'status': 'error', 'data': 'Midi file was not created'}

if __name__ == '__main__':
    uvicorn.run(app, port=8000, host="0.0.0.0")
