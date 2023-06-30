![Basic Pitch Logo](https://user-images.githubusercontent.com/213293/167478083-de988de2-9137-4325-8a5f-ceeb51233753.png)



[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
![PyPI - Python Version](https://img.shields.io/pypi/pyversions/basic-pitch)
![Supported Platforms](https://img.shields.io/badge/platforms-macOS%20%7C%20Windows%20%7C%20Linux-green)


Basic Pitch is a Python library for Automatic Music Transcription (AMT), using lightweight neural network developed by [Spotify's Audio Intelligence Lab](https://research.atspotify.com/audio-intelligence/). It's small, easy-to-use, `pip install`-able and `npm install`-able via its [sibling repo](https://github.com/spotify/basic-pitch-ts).

Basic Pitch may be simple, but it's is far from "basic"! `basic-pitch` is efficient and easy to use, and its multipitch support, its ability to generalize across instruments, and its note accuracy competes with much larger and more resource-hungry AMT systems.

Provide a compatible audio file and basic-pitch will generate a MIDI file, complete with pitch bends. Basic pitch is instrument-agnostic and supports polyphonic instruments, so you can freely enjoy transcription of all your favorite music, no matter what instrument is used.  Basic pitch works best on one instrument at a time.

### Research Paper
This library was released in conjunction with Spotify's publication at [ICASSP 2022](https://2022.ieeeicassp.org/). You can read more about this research in the paper, [A Lightweight Instrument-Agnostic Model for Polyphonic Note Transcription and Multipitch Estimation](https://arxiv.org/abs/2203.09893).

If you use this library in academic research, consider citing it:
```bibtex
@inproceedings{2022_BittnerBRME_LightweightNoteTranscription_ICASSP,
  author= {Bittner, Rachel M. and Bosch, Juan Jos\'e and Rubinstein, David and Meseguer-Brocal, Gabriel and Ewert, Sebastian},
  title= {A Lightweight Instrument-Agnostic Model for Polyphonic Note Transcription and Multipitch Estimation},
  booktitle= {Proceedings of the IEEE International Conference on Acoustics, Speech, and Signal Processing (ICASSP)},
  address= {Singapore},
  year= 2022,
}
```

**Note that we have improved Basic Pitch beyond what was presented in this paper. Therefore, if you use the output of Basic Pitch in academic research,
we recommend that you cite the version of the code that was used.**

### Demo
If, for whatever reason, you're not yet completely inspired, or you're just like so totally over the general vibe and stuff, checkout our snappy demo website, [basicpitch.io](https://basicpitch.io), to experiment with our model on whatever music audio you provide!


## Installation

`basic-pitch` is available via PyPI. To install the current release:

    pip install basic-pitch

To update Basic Pitch to the latest version, add `--upgrade` to the above command.

#### Compatible Environments:
- MacOS, Windows and Ubuntu operating systems
- Python versions 3.7, 3.8, 3.9, 3.10
- **For Mac M1 hardware, we currently only support python version 3.10. Otherwise, we suggest using a virtual machine.**


## Usage

### Model Prediction

#### Command Line Tool

This library offers a command line tool interface. A basic prediction command will generate and save a MIDI file transcription of audio at the `<input-audio-path>` to the `<output-directory>`:

```bash
basic-pitch <output-directory> <input-audio-path>
```

For example: 
```
basic-pitch /output/directory/path /input/audio/path
```

To process more than one audio file at a time:

```bash
basic-pitch <output-directory> <input-audio-path-1> <input-audio-path-2> <input-audio-path-3>
```

Optionally, you may append any of the following flags to your prediction command to save additional formats of the prediction output to the `<output-directory>`:

- `--sonify-midi` to additionally save a `.wav` audio rendering of the MIDI file
- `--save-model-outputs` to additionally save raw model outputs as an NPZ file
- `--save-note-events` to additionally save the predicted note events as a CSV file

To discover more parameter control, run:
```bash
basic-pitch --help
```

#### Programmatic

**predict()**

Import `basic-pitch` into your own Python code and run the [`predict`](basic_pitch/inference.py) functions directly, providing an `<input-audio-path>` and returning the model's prediction results:

```python
from basic_pitch.inference import predict
from basic_pitch import ICASSP_2022_MODEL_PATH

model_output, midi_data, note_events = predict(<input-audio-path>)
```

- `<minimum-frequency>` & `<maximum-frequency>` (*float*s) set the maximum and minimum allowed note frequency, in Hz, returned by the model. Pitch events with frequencies outside of this range will be excluded from the prediction results.
- `model_output` is the raw model inference output
- `midi_data` is the transcribed MIDI data derived from the `model_output`
- `note_events` is a list of note events derived from the `model_output`

**predict() in a loop**

To run prediction within a loop, you'll want to load the model yourself and provide `predict()` with the loaded model object itself to be used for repeated prediction calls, in order to avoid redundant and sluggish model loading.

```python
import tensorflow as tf

from basic_pitch.inference import predict
from basic_pitch import ICASSP_2022_MODEL_PATH

basic_pitch_model = tf.saved_model.load(str(ICASSP_2022_MODEL_PATH))

for x in range():
    ...
    model_output, midi_data, note_events = predict(
        <loop-x-input-audio-path>,
        basic_pitch_model,
    )
    ...
```

**predict_and_save()**

If you would like `basic-pitch` orchestrate the generation and saving of our various supported output file types, you may use [`predict_and_save`](basic_pitch/inference.py) instead of using [`predict`](basic_pitch/inference.py) directly:

```python
from basic_pitch.inference import predict_and_save

predict_and_save(
    <input-audio-path-list>,
    <output-directory>,
    <save-midi>,
    <sonify-midi>,
    <save-model-outputs>,
    <save-notes>,
)
```

where:
   - `<input-audio-path-list>` & `<output-directory>`
        - directory paths for `basic-pitch` to read from/write to.
   - `<save-midi>`
        - *bool* to control generating and saving a MIDI file to the `<output-directory>`
   - `<sonify-midi>`
        - *bool* to control saving a WAV audio rendering of the MIDI file to the `<output-directory>`
   - `<save-model-outputs>`
        - *bool* to control saving the raw model output as a NPZ file to the `<output-directory>`
   - `<save-notes>`
        - *bool* to control saving predicted note events as a CSV file `<output-directory>`



### Model Input

**Supported Audio Codecs**

   `basic-pitch` accepts all sound files that are compatible with its version of [`librosa`](https://librosa.org/doc/latest/index.html), including:

- `.mp3`
- `.ogg`
- `.wav`
- `.flac`
- `.m4a`

**Mono Channel Audio Only**

While you may use stereo audio as an input to our model, at prediction time, the channels of the input will be down-mixed to mono, and then analyzed and transcribed.

**File Size/Audio Length**

This model can process any size or length of audio, but processing of larger/longer audio files could be limited by your machine's available disk space. To process these files, we recommend streaming the audio of the file, processing windows of audio at a time.

**Sample Rate**

Input audio maybe be of any sample rate, however, all audio will be resampled to 22050 Hz before processing.

### VST

Thanks to DamRsn for developing this working VST version of basic-pitch! - https://github.com/DamRsn/NeuralNote


## Contributing

Contributions to `basic-pitch` are welcomed! See [CONTRIBUTING.md](CONTRIBUTING.md) for details.

## Copyright and License
`basic-pitch` is Copyright 2022 Spotify AB.

This software is licensed under the Apache License, Version 2.0 (the "Apache License"). You may choose either license to govern your use of this software only upon the condition that you accept all of the terms of either the Apache License.

You may obtain a copy of the Apache License at:

http://www.apache.org/licenses/LICENSE-2.0


Unless required by applicable law or agreed to in writing, software distributed under the Apache License or the GPL License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the Apache License for the specific language governing permissions and limitations under the Apache License.

