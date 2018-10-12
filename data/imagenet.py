import tensorflow as tf
import numpy as np
from . import dataset


class Dataset(dataset.Dataset):

    def __init__(self, image_size, data_format, filenames, num_epochs, batch_size, buffer_size):

        self.image_size = image_size
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
                    shape=[],
                    dtype=tf.int64,
                    default_value=0
                )
            }
        )

        image = tf.read_file(features["path"])
        image = tf.image.decode_jpeg(image, 3)
        image = tf.image.resize_images(image, self.image_size)
        image -= tf.reshape([123.68, 116.78, 103.94], [1, 1, 3])

        if self.data_format == "channels_first":
            image = tf.transpose(image, [2, 0, 1])

        label = tf.cast(features["label"], tf.int32)

        return image, label
