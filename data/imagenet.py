import tensorflow as tf
import numpy as np
from . import dataset


class Dataset(dataset.Dataset):

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

        image = tf.py_func(np.load, [features["path"]], tf.float32)
        image = tf.reshape(image, [224, 224, 3])
        label = tf.cast(features["label"], tf.int32)

        return image, label
