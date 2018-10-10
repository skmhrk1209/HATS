import tensorflow as tf
from . import dataset
from tensorflow.python.keras.applications import resnet50


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

        image = tf.read_file(features["path"])
        image = tf.image.decode_jpeg(image, 3)
        image = tf.image.resize_images(image, [224, 224])
        #image = tf.py_func(resnet50.preprocess_input, [image], tf.float32)

        label = tf.cast(features["label"], tf.int32)

        return image, label
