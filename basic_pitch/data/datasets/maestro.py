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
import sys
import tempfile
import time
from typing import Any, Dict, List, TextIO, Tuple

import apache_beam as beam
import mirdata

from basic_pitch.data import commandline, pipeline


def read_in_chunks(file_object: TextIO, chunk_size: int = 1024) -> Any:
    """Lazy function (generator) to read a file piece by piece.
    Default chunk size: 1k."""
    while True:
        data = file_object.read(chunk_size)
        if not data:
            break
        yield data


class MaestroInvalidTracks(beam.DoFn):
    DOWNLOAD_ATTRIBUTES = ["audio_path"]

    def __init__(self, source: str) -> None:
        self.source = source

    def setup(self) -> None:
        # Oddly enough we dont want to include the gcs bucket uri.
        # Just the path within the bucket
        self.maestro_remote = mirdata.initialize("maestro", data_home=self.source)
        self.filesystem = beam.io.filesystems.FileSystems()

    def process(self, element: Tuple[str, str], *args: Tuple[Any, Any], **kwargs: Dict[str, Any]) -> Any:
        import tempfile
        import sox

        track_id, split = element
        logging.info(f"Processing (track_id, split): ({track_id}, {split})")

        track_remote = self.maestro_remote.track(track_id)
        with tempfile.TemporaryDirectory() as local_tmp_dir:
            maestro_local = mirdata.initialize("maestro", local_tmp_dir)
            track_local = maestro_local.track(track_id)

            for attribute in self.DOWNLOAD_ATTRIBUTES:
                source = getattr(track_remote, attribute)
                destination = getattr(track_local, attribute)
                os.makedirs(os.path.dirname(destination), exist_ok=True)
                with self.filesystem.open(source) as s, open(destination, "wb") as d:
                    for piece in read_in_chunks(s):
                        d.write(piece)

            # 15 minutes * 60 seconds/minute
            if sox.file_info.duration(track_local.audio_path) >= 15 * 60:
                return None

        yield beam.pvalue.TaggedOutput(split, track_id)


class MaestroToTfExample(beam.DoFn):
    DOWNLOAD_ATTRIBUTES = ["audio_path", "midi_path"]

    def __init__(self, source: str, download: bool):
        self.source = source
        self.download = download

    def setup(self) -> None:
        import apache_beam as beam
        import mirdata

        # Oddly enough we dont want to include the gcs bucket uri.
        # Just the path within the bucket
        self.maestro_remote = mirdata.initialize("maestro", data_home=self.source)
        self.filesystem = beam.io.filesystems.FileSystems()
        if self.download:
            self.maestro_remote.download()

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
            N_FREQ_BINS_NOTES,
            N_FREQ_BINS_CONTOURS,
        )
        from basic_pitch.data import tf_example_serialization

        logging.info(f"Processing {element}")
        batch = []

        for track_id in element:
            track_remote = self.maestro_remote.track(track_id)
            with tempfile.TemporaryDirectory() as local_tmp_dir:
                maestro_local = mirdata.initialize("maestro", local_tmp_dir)
                track_local = maestro_local.track(track_id)

                for attribute in self.DOWNLOAD_ATTRIBUTES:
                    source = getattr(track_remote, attribute)
                    destination = getattr(track_local, attribute)
                    os.makedirs(os.path.dirname(destination), exist_ok=True)
                    with self.filesystem.open(source) as s, open(destination, "wb") as d:
                        # d.write(s.read())
                        for piece in read_in_chunks(s):
                            d.write(piece)

                local_wav_path = f"{track_local.audio_path}_tmp.wav"

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
                contour_indices, contour_values = track_local.notes.to_sparse_index(
                    time_scale, "s", FREQ_BINS_CONTOURS, "hz"
                )

                batch.append(
                    tf_example_serialization.to_transcription_tfexample(
                        track_local.track_id,
                        "maestro",
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
        return [batch]


def create_input_data(source: str) -> List[Tuple[str, str]]:
    import apache_beam as beam

    filesystem = beam.io.filesystems.FileSystems()

    with tempfile.TemporaryDirectory() as tmpdir:
        maestro = mirdata.initialize("maestro", data_home=tmpdir)
        metadata_path = maestro._index["metadata"]["maestro-v2.0.0"][0]
        with filesystem.open(
            os.path.join(source, metadata_path),
        ) as s, open(os.path.join(tmpdir, metadata_path), "wb") as d:
            d.write(s.read())

        return [(track_id, track.split) for track_id, track in maestro.load_tracks().items()]


def main(known_args: argparse.Namespace, pipeline_args: List[str]) -> None:
    time_created = int(time.time())
    destination = commandline.resolve_destination(known_args, time_created)

    # TODO: Remove  or abstract for foss
    pipeline_options = {
        "runner": known_args.runner,
        "job_name": f"maestro-tfrecords-{time_created}",
        "machine_type": "e2-highmem-4",
        "num_workers": 25,
        "disk_size_gb": 128,
        "experiments": ["use_runner_v2", "no_use_multiple_sdk_containers"],
        "save_main_session": True,
        "sdk_container_image": known_args.sdk_container_image,
        "job_endpoint": known_args.job_endpoint,
        "environment_type": "DOCKER",
        "environment_config": known_args.sdk_container_image,
    }
    input_data = create_input_data(known_args.source)
    pipeline.run(
        pipeline_options,
        pipeline_args,
        input_data,
        MaestroToTfExample(known_args.source, download=True),
        MaestroInvalidTracks(known_args.source),
        destination,
        known_args.batch_size,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    commandline.add_default(parser, os.path.basename(os.path.splitext(__file__)[0]))
    commandline.add_split(parser)
    known_args, pipeline_args = parser.parse_known_args(sys.argv)

    main(known_args, pipeline_args)
