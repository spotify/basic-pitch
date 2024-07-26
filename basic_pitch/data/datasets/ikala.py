#!/usr/bin/env python
# encoding: utf-8
#
# Copyright 2024 Spotify AB
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
import random
import sys
import time
from typing import Any, Dict, List, Tuple, Optional

import apache_beam as beam
import mirdata

from basic_pitch.data import commandline, pipeline


class IkalaInvalidTracks(beam.DoFn):
    def process(self, element: Tuple[str, str], *args: Tuple[Any, Any], **kwargs: Dict[str, Any]) -> Any:
        track_id, split = element
        yield beam.pvalue.TaggedOutput(split, track_id)


class IkalaToTfExample(beam.DoFn):
    DOWNLOAD_ATTRIBUTES = ["audio_path", "notes_pyin_path", "f0_path"]

    def __init__(self, source: str, download: bool) -> None:
        self.source = source
        self.download = download

    def setup(self) -> None:
        import apache_beam as beam
        import os
        import mirdata

        self.ikala_remote = mirdata.initialize("ikala", data_home=os.path.join(self.source, "iKala"))
        self.filesystem = beam.io.filesystems.FileSystems()  # TODO: replace with fsspec
        if self.download:
            self.ikala_remote.download()

    def process(self, element: List[str], *args: Tuple[Any, Any], **kwargs: Dict[str, Any]) -> List[Any]:
        import tempfile

        import numpy as np
        import sox

        from basic_pitch.constants import (
            AUDIO_N_CHANNELS,
            AUDIO_SAMPLE_RATE,
            FREQ_BINS_CONTOURS,
            FREQ_BINS_NOTES,
            ANNOTATION_HOP,
            N_FREQ_BINS_CONTOURS,
            N_FREQ_BINS_NOTES,
        )
        from basic_pitch.data import tf_example_serialization

        logging.info(f"Processing {element}")
        batch = []

        for track_id in element:
            track_remote = self.ikala_remote.track(track_id)
            with tempfile.TemporaryDirectory() as local_tmp_dir:
                ikala_local = mirdata.initialize("ikala", local_tmp_dir)
                track_local = ikala_local.track(track_id)

                for attr in self.DOWNLOAD_ATTRIBUTES:
                    source = getattr(track_remote, attr)
                    dest = getattr(track_local, attr)
                    os.makedirs(os.path.dirname(dest), exist_ok=True)
                    with self.filesystem.open(source) as s, open(dest, "wb") as d:
                        d.write(s.read())

                local_wav_path = "{}_tmp.wav".format(track_local.audio_path)

                tfm = sox.Transformer()
                tfm.rate(AUDIO_SAMPLE_RATE)
                tfm.remix({1: [2]})
                tfm.channels(AUDIO_N_CHANNELS)
                tfm.build(track_local.audio_path, local_wav_path)

                duration = sox.file_info.duration(local_wav_path)
                time_scale = np.arange(0, duration + ANNOTATION_HOP, ANNOTATION_HOP)
                n_time_frames = len(time_scale)

                if track_local.notes_pyin is not None:
                    note_indices, note_values = track_local.notes_pyin.to_sparse_index(
                        time_scale, "s", FREQ_BINS_NOTES, "hz"
                    )
                    onset_indices, onset_values = track_local.notes_pyin.to_sparse_index(
                        time_scale, "s", FREQ_BINS_NOTES, "hz", onsets_only=True
                    )
                    note_shape = (n_time_frames, N_FREQ_BINS_NOTES)
                # if there are no notes, return empty note indices
                else:
                    note_indices = []
                    onset_indices = []
                    note_values = []
                    onset_values = []
                    note_shape = (0, 0)

                contour_indices, contour_values = track_local.f0.to_sparse_index(
                    time_scale, "s", FREQ_BINS_CONTOURS, "hz"
                )

                batch.append(
                    tf_example_serialization.to_transcription_tfexample(
                        track_id,
                        "ikala",
                        local_wav_path,
                        note_indices,
                        note_values,
                        onset_indices,
                        onset_values,
                        contour_indices,
                        contour_values,
                        note_shape,
                        (n_time_frames, N_FREQ_BINS_CONTOURS),
                    )
                )
        return [batch]


def create_input_data(train_percent: float, seed: Optional[int] = None) -> List[Tuple[str, str]]:
    assert train_percent < 1.0, "Don't over allocate the data!"

    if seed:
        random.seed(seed)

    ikala = mirdata.initialize("ikala")
    track_ids = ikala.track_ids
    random.shuffle(track_ids)

    def determine_split(index: int) -> str:
        return "train" if index < len(track_ids) * train_percent else "validation"

    return [(track_id, determine_split(i)) for i, track_id in enumerate(track_ids)]


def main(known_args: argparse.Namespace, pipeline_args: List[str]) -> None:
    time_created = int(time.time())
    destination = commandline.resolve_destination(known_args, time_created)

    pipeline_options = {
        "runner": known_args.runner,
        "job_name": f"ikala-tfrecords-{time_created}",
        "machine_type": "e2-standard-4",
        "num_workers": 25,
        "disk_size_gb": 128,
        "experiments": ["use_runner_v2", "no_use_multiple_sdk_containers"],
        "save_main_session": True,
        "sdk_container_image": known_args.sdk_container_image,
        "job_endpoint": known_args.job_endpoint,
        "environment_type": "DOCKER",
        "environment_config": known_args.sdk_container_image,
    }
    input_data = create_input_data(known_args.train_percent, known_args.split_seed)
    pipeline.run(
        pipeline_options,
        pipeline_args,
        input_data,
        IkalaToTfExample(known_args.source, download=True),
        IkalaInvalidTracks(known_args.source),
        destination,
        known_args.batch_size,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    commandline.add_default(parser, os.path.basename(os.path.splitext(__file__)[0]))
    commandline.add_split(parser)
    known_args, pipeline_args = parser.parse_known_args(sys.argv)

    main(known_args, pipeline_args)
