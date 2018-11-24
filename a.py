import tensorflow as tf
import numpy as np
import functools
import os
import glob
import cv2
from tqdm import trange


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


dataset = Dataset(filenames=[filename for filename in glob.glob("/home/sakuma/data/fsns/*") if "test" in filename or "validation" in filename])
next_element = dataset.get_next()

with tf.Session() as sess:

    for i in trange(1100000):

        image, label = sess.run(next_element)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        label = label.decode("utf-8")
        label = label.replace(" ", "_")
        cv2.imwrite("/home/sakuma/data/fsns_/{}/{}_{}.jpg".format("test", i, label), image)
