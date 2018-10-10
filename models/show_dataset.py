import tensorflow as tf
import cv2

def parse(example):

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

    return image

dataset = tf.data.TFRecordDataset("train.tfrecord")
dataset = dataset.map(parse)
get_next = dataset.make_one_shot_iterator().get_next()

with tf.Session() as session:

    while True:
        image = session.run(get_next)

        cv2.imshow("", image)
        cv2.waitKey(1000)