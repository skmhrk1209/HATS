import tensorflow as tf
import numpy as np
import functools


class Dataset(object):

    def __init__(self, filenames, num_epochs, batch_size, buffer_size, num_cpus,
                 image_size, data_format, sequence_length, string_length):

        self.image_size = image_size
        self.data_format = data_format
        self.sequence_length = sequence_length
        self.string_length = string_length

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
            num_parallel_calls=num_cpus
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
                    dtype=tf.string,
                    default_value=""
                ),
                "label": tf.FixedLenFeature(
                    shape=[self.sequence_length * self.string_length],
                    dtype=tf.int64,
                    default_value=[62] * (self.sequence_length * self.string_length)
                )
            }
        )

        image = tf.read_file(features["path"])
        image = tf.image.decode_jpeg(image, 3)
        image = tf.image.convert_image_dtype(image, tf.float32)
        image.set_shape([256, 256, 3])

        if self.image_size:
            image = tf.image.resize_images(image, self.image_size)

        if self.data_format == "channels_first":
            image = tf.transpose(image, [2, 0, 1])

        label = tf.cast(features["label"], tf.int32)
        label = tf.reshape(label, [self.sequence_length, self.string_length])

        return {"image": image}, label

    def get_next(self):

        return self.iterator.get_next()
