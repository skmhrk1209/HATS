import tensorflow as tf
import numpy as np
import functools
import glob
import os
from algorithms import *


class Dataset(object):

    def __init__(self, filenames, num_epochs, batch_size, random_seed,
                 sequence_lengths, image_size, data_format, encoding):

        self.dataset = tf.data.TFRecordDataset(
            filenames=filenames,
            num_parallel_reads=os.cpu_count()
        )
        self.dataset = self.dataset.shuffle(
            buffer_size=sum([
                len(list(tf.python_io.tf_record_iterator(filename)))
                for filename in filenames
            ]),
            seed=random_seed,
            reshuffle_each_iteration=True
        )
        self.dataset = self.dataset.repeat(
            count=num_epochs
        )
        self.dataset = self.dataset.map(
            map_func=functools.partial(
                self.parse,
                sequence_lengths=sequence_lengths,
                image_size=image_size,
                data_format=data_format,
                encoding=encoding
            ),
            num_parallel_calls=os.cpu_count()
        )
        self.dataset = self.dataset.batch(
            batch_size=batch_size,
            drop_remainder=True
        )
        self.dataset = self.dataset.prefetch(
            buffer_size=1
        )

    def __call__(self):

        return self.dataset.make_one_shot_iterator().get_next()

    def parse(self, example, sequence_lengths, image_size, data_format, encoding):

        features = tf.parse_single_example(
            serialized=example,
            features={
                "path": tf.FixedLenFeature(
                    shape=[],
                    dtype=tf.string
                ),
                "label": tf.FixedLenFeature(
                    shape=[np.prod(sequence_lengths)],
                    dtype=tf.int64
                )
            }
        )

        image = tf.read_file(features["path"])
        if encoding == "jpeg":
            image = tf.image.decode_jpeg(image, 3)
        elif encoding == "png":
            image = tf.image.decode_png(image, 3)
        image = tf.image.convert_image_dtype(image, tf.float32)
        if image_size:
            image = tf.image.resize_images(image, image_size)
        if data_format == "channels_first":
            image = tf.transpose(image, [2, 0, 1])

        label = tf.cast(features["label"], tf.int32)
        label = tf.reshape(label, sequence_lengths)

        return image, label
