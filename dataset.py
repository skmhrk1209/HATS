import tensorflow as tf
import numpy as np
import functools
import os


class Dataset(object):

    def __init__(self, filenames, num_epochs, batch_size, buffer_size,
                 image_size, data_format, sequence_length, string_length):

        self.dataset = tf.data.TFRecordDataset(filenames)
        self.dataset = self.dataset.shuffle(
            buffer_size=buffer_size,
            reshuffle_each_iteration=True
        )
        self.dataset = self.dataset.repeat(num_epochs)
        self.dataset = self.dataset.map(
            map_func=functools.partial(
                self.parse,
                image_size=image_size,
                data_format=data_format,
                sequence_length=sequence_length,
                string_length=string_length
            ),
            num_parallel_calls=os.cpu_count()
        )
        self.dataset = self.dataset.batch(batch_size)
        self.dataset = self.dataset.prefetch(1)
        self.iterator = self.dataset.make_one_shot_iterator()

    def parse(self, example, image_size, data_format, sequence_length, string_length):

        features = tf.parse_single_example(
            serialized=example,
            features={
                "path": tf.FixedLenFeature(
                    shape=[],
                    dtype=tf.string
                ),
                "label": tf.FixedLenFeature(
                    shape=[sequence_length * string_length],
                    dtype=tf.int64
                )
            }
        )

        image = tf.read_file(features["path"])
        image = tf.image.decode_jpeg(image, 3)
        image = tf.image.convert_image_dtype(image, tf.float32)
        image.set_shape([256, 256, 3])

        if image_size:
            image = tf.image.resize_images(image, image_size)

        if data_format == "channels_first":
            image = tf.transpose(image, [2, 0, 1])

        label = tf.cast(features["label"], tf.int32)
        label = tf.reshape(label, [sequence_length, string_length])

        return {"image": image}, label

    def get_next(self):

        return self.iterator.get_next()
