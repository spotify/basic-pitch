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

import csv
import os

from itertools import islice

import apache_beam as beam
import numpy as np
import tensorflow as tf

from apache_beam.testing.test_pipeline import TestPipeline

from basic_pitch.data.tf_example_serialization import bytes_feature, int64_feature
from basic_pitch.data.pipeline import (
    Batch,
    WriteBatchToTfRecord,
    transcription_dataset_writer,
)


def test_batch() -> None:
    element_size = 10
    batch_size = 3

    element = np.random.random([element_size])
    batcher = Batch(batch_size)
    batches = list(batcher.process(element))
    for batch in batches[:-1]:
        assert len(batch) == element_size // batch_size

    assert len(batches[-1]) == element_size % batch_size


def test_write_batch_to_tf_record(tmpdir: str) -> None:
    data = np.random.random([10, 10])
    features = {f"feat{i}": bytes_feature(tf.io.serialize_tensor(vec)) for i, vec in enumerate(list(data))}
    example = tf.train.Example(features=tf.train.Features(feature=features))

    writer = WriteBatchToTfRecord(tmpdir)
    writer.process([example])
    assert len(os.listdir(tmpdir)) == 1
    dataset = tf.data.TFRecordDataset([os.path.join(tmpdir, f) for f in os.listdir(tmpdir)])

    for d in dataset:
        ex = tf.train.Example()
        ex.ParseFromString(d.numpy())

    assert ex == example


def test_transcription_dataset_writer(tmpdir: str) -> None:
    input_data = [(f"track{i}", ("train", "test", "validation")[i % 3]) for i in range(9)] + [("foo", "bar")]
    batch_size = 2

    with TestPipeline() as p:
        transcription_dataset_writer(
            p=p,
            input_data=input_data,
            to_tf_example=lambda el: [tf.train.Example(features=tf.train.Features(feature={"foo": int64_feature(1)}))],
            filter_invalid_tracks=lambda el: [beam.pvalue.TaggedOutput(el[1], el[0])],
            destination=tmpdir,
            batch_size=batch_size,
        )

    for i, split in enumerate(["train", "test", "validation"]):
        assert os.path.exists(os.path.join(tmpdir, split, "index.csv"))
        with open(os.path.join(tmpdir, split, "index.csv")) as fp:
            reader = csv.reader(fp, delimiter=",")
            for row in islice(reader, 1, None):
                assert int(row[0][-1]) % 3 == i
