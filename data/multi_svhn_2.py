import tensorflow as tf
import numpy as np
from . import dataset


class Dataset(dataset.Dataset):

    def __init__(self, filenames, num_epochs, batch_size, buffer_size,
                 data_format, image_size, digits_length, sequence_length):

        self.data_format = data_format
        self.image_size = image_size
        self.digits_length = digits_length
        self.sequence_length = sequence_length

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
                "labels": tf.FixedLenFeature(
                    shape=[self.sequence_length * self.digits_length],
                    dtype=tf.int64,
                    default_value=[10] * (self.sequence_length * self.digits_length)
                )
            }
        )

        image = tf.read_file(features["path"])
        image = tf.image.decode_png(image, 3)
        image = tf.image.convert_image_dtype(image, tf.float32)
        image = tf.image.resize_images(image, self.image_size)

        if self.data_format == "channels_first":
            image = tf.transpose(image, [2, 0, 1])

        labels = tf.cast(features["labels"], tf.int32)

        return {"images": image}, labels
