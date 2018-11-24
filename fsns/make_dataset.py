import tensorflow as tf
import glob
import cv2

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
        label = features["label"]

        return image, label

train_filenames = [f for f in glob.glob("/home/sakuma/data/fsns_raw/*") if "train" in f]

dataset = tf.data.TFRecordDataset(train_filenames)
dataset = dataset.map(parse)
next_elem = dataset.make_one_shot_iterator().get_next()

with tf.Session() as sess:

    image, label = sess.run(next_elem)

    print(label)
    cv2.imshow("", image)
    cv2.waitKey()