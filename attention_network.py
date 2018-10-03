from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np


class AttentionNetwork(object):

    def __init__(self, conv_params, deconv_params, bottleneck_units, data_format):

        self.conv_params = conv_params
        self.deconv_params = deconv_params
        self.bottleneck_units = bottleneck_units
        self.data_format = data_format

    def __call__(self, inputs, training, name="attention_network", reuse=None):

        with tf.variable_scope(name, reuse=reuse):

            for i, conv_param in enumerate(self.conv_params):

                with tf.variable_scope("conv_block_{}".format(i)):

                    inputs = tf.layers.conv2d(
                        inputs=inputs,
                        filters=conv_param.filters,
                        kernel_size=conv_param.kernel_size,
                        strides=conv_param.strides,
                        padding="same"
                    )

                    inputs = tf.nn.relu(inputs)

            shape = inputs.shape.as_list()

            with tf.variable_scope("bottleneck_block"):

                inputs = tf.layers.flatten(inputs)

                inputs = tf.layers.dense(
                    inputs=inputs,
                    units=self.bottleneck_units
                )

                inputs = tf.nn.relu(inputs)

            with tf.variable_scope("projection_block"):

                inputs = tf.layers.dense(
                    inputs=inputs,
                    units=np.prod(shape[1:])
                )

                inputs = tf.nn.relu(inputs)

                inputs = tf.reshape(
                    tensor=inputs,
                    shape=[-1] + shape[1:]
                )

            for i, deconv_param in enumerate(self.deconv_params):

                with tf.variable_scope("deconv_block_{}".format(i)):

                    inputs = tf.layers.conv2d_transpose(
                        inputs=inputs,
                        filters=deconv_param.filters,
                        kernel_size=deconv_param.kernel_size,
                        strides=deconv_param.strides,
                        padding="same"
                    )

                    if i == len(self.deconv_params) - 1:
                        inputs = tf.nn.sigmoid(inputs)

                    else:
                        inputs = tf.nn.relu(inputs)

            return inputs
