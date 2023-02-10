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

from basic_pitch.note_creation import drop_overlapping_pitch_bends


def test_drop_overlapping_pitch_bends() -> None:
    # events are: (start, end, pitch, amplitude, pitch_bend)
    note_events_with_pitch_bends = [
        (0.0, 0.1, 60, 1.0, None),
        (2.0, 2.1, 62, 1.0, [0, 1, 2]),  # οverlaps w next
        (2.0, 2.1, 64, 1.0, [0, 1, 2]),  # overlaps w prev
        (1.0, 1.1, 65, 1.0, [0, 1, 2]),
        (1.1, 1.2, 67, 1.0, [0, 1, 2]),
        (3.0, 3.2, 69, 1.0, [0, 1, 2]),  # overlaps w next
        (3.1, 3.3, 71, 1.0, [0, 1, 2]),  # overlaps w prev
        (5.0, 5.1, 72, 1.0, [0, 1, 2]),  # overlaps w next
        (5.0, 5.2, 74, 1.0, [0, 1, 2]),  # overlaps w prev
        (4.0, 4.2, 76, 1.0, [0, 1, 2]),  # overlaps w next
        (4.1, 4.2, 77, 1.0, [0, 1, 2]),  # overlaps w prev
    ]
    expected = [
        (0.0, 0.1, 60, 1.0, None),
        (2.0, 2.1, 62, 1.0, None),  # overlaps w next
        (2.0, 2.1, 64, 1.0, None),  # overlaps w prev
        (1.0, 1.1, 65, 1.0, [0, 1, 2]),
        (1.1, 1.2, 67, 1.0, [0, 1, 2]),
        (3.0, 3.2, 69, 1.0, None),  # overlaps w next
        (3.1, 3.3, 71, 1.0, None),  # overlaps w prev
        (5.0, 5.1, 72, 1.0, None),  # overlaps w next
        (5.0, 5.2, 74, 1.0, None),  # overlaps w prev
        (4.0, 4.2, 76, 1.0, None),  # overlaps w next
        (4.1, 4.2, 77, 1.0, None),  # overlaps w prev
    ]
    result = drop_overlapping_pitch_bends(note_events_with_pitch_bends)
    assert sorted(result) == sorted(expected)
