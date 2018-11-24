import tensorflow as tf
import glob
import cv2
import itertools
from tqdm import trange


def parse(example):

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


train_filenames = [f for f in glob.glob("/home/sakuma/data/fsns_raw/*") if "test" in f or "validation" in f]

dataset = tf.data.TFRecordDataset(train_filenames)
dataset = dataset.map(parse)
next_elem = dataset.make_one_shot_iterator().get_next()

with tf.Session() as sess:

    for i in trange(40000):

        image, label = sess.run(next_elem)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        cv2.imwrite("/home/sakuma/data/fsns/test/{} {}.jpg".format(i, label.decode("utf-8")), image)
