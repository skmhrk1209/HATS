import tensorflow as tf
import numpy as np
import functools
import glob
import os
from algorithms import *


def parse_example(example, sequence_lengths, encoding, image_size, data_format):

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

    image = tf.cast(features["path"], tf.string)
    image = tf.read_file(image)
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


def input_fn(filenames, batch_size, num_epochs, shuffle,
             sequence_lengths, encoding, image_size, data_format):

    dataset = tf.data.TFRecordDataset(
        filenames=filenames,
        # num_parallel_reads=os.cpu_count()
    )
    if shuffle:
        dataset = dataset.shuffle(
            buffer_size=sum([
                len(list(tf.io.tf_record_iterator(filename)))
                for filename in filenames
            ]),
            reshuffle_each_iteration=True
        )
    dataset = dataset.repeat(count=num_epochs)
    dataset = dataset.map(
        map_func=functools.partial(
            parse_example,
            sequence_lengths=sequence_lengths,
            encoding=encoding,
            image_size=image_size,
            data_format=data_format
        ),
        num_parallel_calls=os.cpu_count()
    )
    dataset = dataset.batch(batch_size=batch_size)
    dataset = dataset.prefetch(buffer_size=1)

    return dataset.make_one_shot_iterator().get_next()
