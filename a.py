import tensorflow as tf
from algorithms import *

labels = tf.random_uniform([2, 3, 5], maxval=10, dtype=tf.int32)


with tf.Session() as sess:

    for t in sess.run([labels, tf.reshape(labels, [-1, 5])]):
        print(t)
