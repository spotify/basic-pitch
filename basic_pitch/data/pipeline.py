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

import logging
import os
import uuid
from typing import Any, Dict, List, Tuple, Callable, Union

import apache_beam as beam
import tensorflow as tf
from apache_beam.options.pipeline_options import PipelineOptions


# Beacase beam.GroupIntoBatches isn't supported as of 2.29
class Batch(beam.DoFn):
    def __init__(self, batch_size: int) -> None:
        self.batch_size = batch_size

    def process(self, element: List[Any], *args: Tuple[Any, Any], **kwargs: Dict[str, Any]) -> Any:
        for i in range(0, len(element), self.batch_size):
            yield element[i : i + self.batch_size]


class WriteBatchToTfRecord(beam.DoFn):
    def __init__(self, destination: str) -> None:
        self.destination = destination

    def process(self, element: Any, *args: Tuple[Any, Any], **kwargs: Dict[str, Any]) -> None:
        if not isinstance(element, list):
            element = [element]

        logging.info(f"Writing to file batch of length {len(element)}")
        # hopefully uuids are unique enough
        with tf.io.TFRecordWriter(os.path.join(self.destination, f"{uuid.uuid4()}.tfrecord")) as writer:
            for example in element:
                writer.write(example.SerializeToString())


def transcription_dataset_writer(
    p: beam.Pipeline,
    input_data: List[Tuple[str, str]],
    to_tf_example: Union[beam.DoFn, Callable[[List[Any]], Any]],
    filter_invalid_tracks: beam.PTransform,
    destination: str,
    batch_size: int,
) -> None:
    valid_track_ids = (
        p
        | "Create PCollection of track IDS" >> beam.Create(input_data)
        | "Remove invalid track IDs"
        >> beam.ParDo(filter_invalid_tracks).with_outputs(
            "train",
            "test",
            "validation",
        )
    )
    for split in ["train", "test", "validation"]:
        (
            getattr(valid_track_ids, split)
            | f"Batch {split}" >> beam.BatchElements(min_batch_size=batch_size, max_batch_size=batch_size)
            | f"Reshuffle {split}" >> beam.Reshuffle()  # To prevent fuses
            | f"Create tf.Example {split} batch" >> beam.ParDo(to_tf_example)
            | f"Write {split} batch to tfrecord" >> beam.ParDo(WriteBatchToTfRecord(os.path.join(destination, split)))
        )
        getattr(valid_track_ids, split) | f"Write {split} index file" >> beam.io.textio.WriteToText(
            os.path.join(destination, split, "index.csv"),
            num_shards=1,
            header="track_id",
            shard_name_template="",
        )


def run(
    pipeline_options: Dict[str, str],
    pipeline_args: List[str],
    input_data: List[Tuple[str, str]],
    to_tf_example: beam.DoFn,
    filter_invalid_tracks: beam.DoFn,
    destination: str,
    batch_size: int,
) -> None:
    logging.info(f"pipeline_options = {pipeline_options}")
    logging.info(f"pipeline_args = {pipeline_args}")
    with beam.Pipeline(options=PipelineOptions(flags=pipeline_args, **pipeline_options)) as p:
        transcription_dataset_writer(p, input_data, to_tf_example, filter_invalid_tracks, destination, batch_size)
