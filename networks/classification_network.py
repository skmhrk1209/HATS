import tensorflow as tf
import numpy as np


class ClassificationNetwork(object):

    def __init__(self, dense_params, num_classes, data_format):

        self.dense_params = dense_params
        self.num_classes = num_classes
        self.data_format = data_format

    def __call__(self, inputs, training, name="classification_network", reuse=None):

        with tf.variable_scope(name, reuse=reuse):

            for i, dense_param in enumerate(self.dense_params):

                with tf.variable_scope("dense_block_{}".format(i)):

                    inputs = tf.layers.dense(
                        inputs=inputs,
                        units=dense_param.units
                    )

                    inputs = tf.nn.relu(inputs)

            with tf.variable_scope("logit_block"):

                inputs = tf.layers.dense(
                    inputs=inputs,
                    units=self.num_classes
                )

            return inputs
