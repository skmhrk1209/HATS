import tensorflow as tf
import numpy as np
from . import base


class Dataset(base.Dataset):

    def __init__(self, filenames, num_epochs, batch_size, buffer_size,
                 image_size, string_length, data_format):

        self.image_size = image_size
        self.string_length = string_length
        self.data_format = data_format

        super(Dataset, self).__init__(filenames, num_epochs, batch_size, buffer_size)

    def parse(self, example):

        features = tf.parse_single_example(
            serialized=example,
            features={
                "path": tf.FixedLenFeature(
                    shape=[],
                    dtype=tf.string,
                    default_value=""
                ),
                "label": tf.FixedLenFeature(
                    shape=[self.string_length],
                    dtype=tf.int64,
                    default_value=[62] * self.string_length
                )
            }
        )

        image = tf.read_file(features["path"])
        image = tf.image.decode_png(image, 3)
        image = tf.image.convert_image_dtype(image, tf.float32)

        if self.image_size:
            image = tf.image.resize_images(image, self.image_size)

        if self.data_format == "channels_first":
            image = tf.transpose(image, [2, 0, 1])

        label = tf.cast(features["label"], tf.int32)

        return {"image": image}, label
