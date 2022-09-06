import os
import tempfile

import apache_beam as beam
import tensorflow as tf
from apache_beam.testing.test_pipeline import TestPipeline
from apache_beam.testing.util import assert_that, equal_to

from spotify_audio_to_midi.dataset import pipeline


def test_write_batch_to_tf_record():
    split = "a"
    example_protos = [
        [
            tf.train.Example(
                features=tf.train.Features(feature={"a": tf.train.Feature(int64_list=tf.train.Int64List(value=[i]))})
            )
            for i in range(j, j + 10)
        ]
        for j in range(2)
    ]
    with tempfile.TemporaryDirectory() as tmpdir:
        os.mkdir(os.path.join(tmpdir, split))
        with TestPipeline() as p:
            p | beam.Create(example_protos) | beam.ParDo(pipeline.WriteBatchToTfRecord(os.path.join(tmpdir, split)))
        with TestPipeline() as p:
            output = (
                p
                | beam.io.tfrecordio.ReadFromTFRecord(
                    os.path.join(os.path.join(tmpdir, split), "*.tfrecord"),
                    coder=beam.coders.ProtoCoder(tf.train.Example),
                )
                | beam.Map(lambda example: example.features.feature["a"].int64_list.value[0])
            )
            assert_that(output, equal_to(sorted([i for j in range(2) for i in range(j, j + 10)])))


def test_transcription_dataset_writer():
    class FilterInvalidTracks(beam.DoFn):
        def process(self, element):
            split, value = element
            if split not in ["train", "test", "validation"]:
                return

            yield beam.pvalue.TaggedOutput(split, value)

    class ToTfRecord(beam.DoFn):
        def __init__(self, destination):
            self.destination = destination

        def process(self, element):
            yield [
                tf.train.Example(
                    features=tf.train.Features(
                        feature={"a": tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))}
                    )
                )
                for value in element
            ]

    input_data = [(split, i) for split in ["train", "test", "validation", "blahblabh"] for i in range(50)]
    batch_size = 2
    with tempfile.TemporaryDirectory() as tmpdir:
        for split in ["train", "test", "validation"]:
            os.mkdir(os.path.join(tmpdir, split))

        with TestPipeline() as p:
            pipeline.transcription_dataset_writer(
                p, input_data, ToTfRecord(tmpdir), FilterInvalidTracks(), tmpdir, batch_size
            )
        for split in ["train", "test", "validation"]:
            with TestPipeline() as p:
                output = (
                    p
                    | f"Read {split}"
                    >> beam.io.tfrecordio.ReadFromTFRecord(
                        os.path.join(tmpdir, split, "*.tfrecord"),
                        coder=beam.coders.ProtoCoder(tf.train.Example),
                    )
                    | f"Parse {split}" >> beam.Map(lambda example: example.features.feature["a"].int64_list.value[0])
                )
                assert_that(output, equal_to(list(range(50))))

            # Now confirm that each file only has two records!
            for record in os.listdir(os.path.join(tmpdir, split)):
                if record.endswith("tfrecord"):
                    assert (
                        len(list(tf.data.TFRecordDataset(os.path.join(tmpdir, split, record)).as_numpy_iterator())) == 2
                    )
