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

import argparse
import logging
import os
import os.path as op
import sys
import time
from typing import List, Tuple

import apache_beam as beam
import mirdata

from basic_pitch.dataset import commandline, pipeline


class SlakhFilterInvalidTracks(beam.DoFn):
    DOWNLOAD_ATTRIBUTES = ["audio_path", "metadata_path", "midi_path"]

    def __init__(self, source: str):
        self.source = source

    def setup(self):
        import apache_beam as beam
        import os
        import mirdata

        self.slakh_remote = mirdata.initialize("slakh", data_home=os.path.join(self.source, "slakh"))
        self.filesystem = beam.io.filesystems.FileSystems()

    def process(self, element: Tuple[str, str]):
        import tempfile

        import apache_beam as beam
        import sox

        from basic_pitch.constants import (
            AUDIO_N_CHANNELS,
            AUDIO_SAMPLE_RATE,
        )

        track_id, split = element
        if split == "omitted":
            return

        logging.info(f"Processing (track_id, split): ({track_id}, {split})")

        track_remote = self.slakh_remote.track(track_id)

        with tempfile.TemporaryDirectory() as local_tmp_dir:
            slakh_local = mirdata.initialize("slakh", local_tmp_dir)
            track_local = slakh_local.track(track_id)

            for attr in self.DOWNLOAD_ATTRIBUTES:
                source = getattr(track_remote, attr)
                dest = getattr(track_local, attr)
                if not dest:
                    return None
                logging.info(f"Downloading {attr} from {source} to {dest}")
                os.makedirs(os.path.dirname(dest), exist_ok=True)
                with self.filesystem.open(source) as s, open(dest, "wb") as d:
                    d.write(s.read())

            if track_local.is_drum:
                return None

            local_wav_path = "{}_tmp.wav".format(track_local.audio_path)
            tfm = sox.Transformer()
            tfm.rate(AUDIO_SAMPLE_RATE)
            tfm.channels(AUDIO_N_CHANNELS)
            try:
                tfm.build(track_local.audio_path, local_wav_path)
            except Exception as e:
                logging.info(f"Could not process {local_wav_path}. Exception: {e}")
                return None

            # if there are no notes, skip this track
            if track_local.notes is None or len(track_local.notes.intervals) == 0:
                return

            yield beam.pvalue.TaggedOutput(split, track_id)


class SlakhToTfExample(beam.DoFn):
    DOWNLOAD_ATTRIBUTES = ["audio_path", "metadata_path", "midi_path"]

    def __init__(self, source: str):
        self.source = source

    def setup(self):
        import apache_beam as beam
        import os
        import mirdata

        self.slakh_remote = mirdata.initialize("slakh", data_home=os.path.join(self.source, "slakh"))
        self.filesystem = beam.io.filesystems.FileSystems()

    def process(self, element: List[str]):
        import tempfile

        import numpy as np
        import sox

        from basic_pitch.constants import (
            AUDIO_N_CHANNELS,
            AUDIO_SAMPLE_RATE,
            FREQ_BINS_CONTOURS,
            FREQ_BINS_NOTES,
            ANNOTATION_HOP,
            N_FREQ_BINS_NOTES,
            N_FREQ_BINS_CONTOURS,
        )
        from basic_pitch.dataset import tf_example_serialization

        logging.info(f"Processing {element}")

        batch = []

        for track_id in element:
            track_remote = self.slakh_remote.track(track_id)

            with tempfile.TemporaryDirectory() as local_tmp_dir:
                slakh_local = mirdata.initialize("slakh", local_tmp_dir)
                track_local = slakh_local.track(track_id)

                for attr in self.DOWNLOAD_ATTRIBUTES:
                    source = getattr(track_remote, attr)
                    dest = getattr(track_local, attr)
                    logging.info(f"Downloading {attr} from {source} to {dest}")
                    os.makedirs(os.path.dirname(dest), exist_ok=True)
                    with self.filesystem.open(source) as s, open(dest, "wb") as d:
                        d.write(s.read())

                local_wav_path = "{}_tmp.wav".format(track_local.audio_path)
                tfm = sox.Transformer()
                tfm.rate(AUDIO_SAMPLE_RATE)
                tfm.channels(AUDIO_N_CHANNELS)
                tfm.build(track_local.audio_path, local_wav_path)

                duration = sox.file_info.duration(local_wav_path)
                time_scale = np.arange(0, duration + ANNOTATION_HOP, ANNOTATION_HOP)
                n_time_frames = len(time_scale)

                note_indices, note_values = track_local.notes.to_sparse_index(time_scale, "s", FREQ_BINS_NOTES, "hz")
                onset_indices, onset_values = track_local.notes.to_sparse_index(
                    time_scale, "s", FREQ_BINS_NOTES, "hz", onsets_only=True
                )
                contour_indices, contour_values = track_local.multif0.to_sparse_index(
                    time_scale, "s", FREQ_BINS_CONTOURS, "hz"
                )

                batch.append(
                    tf_example_serialization.to_transcription_tfexample(
                        track_id,
                        "slakh",
                        local_wav_path,
                        note_indices,
                        note_values,
                        onset_indices,
                        onset_values,
                        contour_indices,
                        contour_values,
                        (n_time_frames, N_FREQ_BINS_NOTES),
                        (n_time_frames, N_FREQ_BINS_CONTOURS),
                    )
                )
        logging.info(f"Finished processing batch of length {len(batch)}")
        return [batch]


def create_input_data() -> List[Tuple[str, str]]:
    slakh = mirdata.initialize("slakh")
    slakh.download()

    return [(track_id, track.data_split) for track_id, track in slakh.load_tracks().items()]


def main(known_args, pipeline_args):
    time_created = int(time.time())
    destination = commandline.resolve_destination(known_args, "slakh", time_created)

    pipeline_options = {
        "runner": known_args.runner,
        "job_name": f"slakh-tfrecords-{time_created}",
        "machine_type": "e2-standard-4",
        "num_workers": 25,
        "disk_size_gb": 128,
        "experiments": ["use_runner_v2"],
        "save_main_session": True,
        "worker_harness_container_image": known_args.worker_harness_container_image,
    }
    input_data = create_input_data()
    pipeline.run(
        pipeline_options,
        input_data,
        SlakhToTfExample(known_args.source),
        SlakhFilterInvalidTracks(known_args.source),
        destination,
        known_args.batch_size,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    commandline.add_default(parser, op.basename(op.splittext(__file__)[0]))
    commandline.add_split(parser)
    known_args, pipeline_args = parser.parse_known_args()  # sys.argv)

    main(known_args, pipeline_args)
