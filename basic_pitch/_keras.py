#!/usr/bin/env python
# encoding: utf-8
#
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

"""Keras 2 compatibility shim.

TensorFlow >= 2.16 (required for Python >= 3.12) ships Keras 3 as ``tf.keras``. Basic Pitch's model and
layer code is written against Keras 2, which is provided for those TensorFlow versions by the ``tf-keras``
package. Import ``keras`` from here instead of using ``tf.keras`` so the Keras 2 API is used regardless of
the TensorFlow version or of the order in which ``tensorflow`` and ``basic_pitch`` are imported.
"""

try:
    import tf_keras as keras
except ImportError:
    from tensorflow import keras

__all__ = ["keras"]
