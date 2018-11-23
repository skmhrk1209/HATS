import tensorflow as tf
import numpy as np
import functools
import os
import glob
import cv2

class Dataset(object):

    def __init__(self, filenames):

        self.dataset = tf.data.TFRecordDataset(filenames)
        self.dataset = self.dataset.map(self.parse)
        self.iterator = self.dataset.make_one_shot_iterator()

    def parse(self, example):

        features = tf.parse_single_example(
            serialized=example,
            features={
                "image/encoded": tf.FixedLenFeature(
                    shape=[],
                    dtype=tf.string
                ),
                "image/text": tf.FixedLenFeature(
                    shape=[],
                    dtype=tf.string
                )
            }
        )

        image = tf.image.decode_png(features["image/encoded"], 3)
        label = features["image/text"]

        return image, label

    def get_next(self):

        return self.iterator.get_next()

dataset = Dataset(filenames=[filename for filename in glob.glob("/home/sakuma/data/fsns/*") if "train" in filename])
next_element = dataset.get_next()

with tf.Session() as sess:

    while True:

        image, label = sess.run(next_element)

        cv2.imshow("", image)
        cv2.waitKey()
