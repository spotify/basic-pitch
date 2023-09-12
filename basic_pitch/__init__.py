#!/usr/bin/env python
# encoding: utf-8
#
# Copyright 2022 Spotify AB
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import pathlib

__author__ = "Spotify"
__version__ = "0.3.0"
__email__ = "basic-pitch@spotify.com"
__demowebsite__ = "https://basicpitch.io"
__description__ = "Basic Pitch, a lightweight yet powerful audio-to-MIDI converter with pitch bend detection."
__url__ = "https://github.com/spotify/basic-pitch"

ICASSP_2022_MODEL_PATH = pathlib.Path(__file__).parent / "saved_models/icassp_2022/nmp"
