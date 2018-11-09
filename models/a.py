import tensorflow as tf
import numpy as np
from data.multi_mjsynth import Dataset

label = Dataset(
    filenames="multi_mjsynth_train.tfrecord",
    num_epochs=1,
    batch_size=128,
    buffer_size=1,
    data_format="channels_last",
    image_size=[256, 256],
    sequence_length=4,
    string_length=10
).get_next()[1]

length = tf.shape(tf.where(tf.not_equal(label, 62)))[0]

with tf.Session() as sess:
    for _ in range(100):
        print(sess.run(length))
