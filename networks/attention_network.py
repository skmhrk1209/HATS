import tensorflow as tf
import numpy as np
from . import ops
from . import rnn_cell
from algorithms import *
from itertools import *


def static_rnn(cell, inputs, initial_state):

    return list(accumulate([initial_state] + inputs, lambda hiddens, inputs: cell(inputs, hiddens)[0]))[1:]


class AttentionNetwork(object):

    def __init__(self, conv_params, rnn_params, deconv_params, data_format):

        self.conv_params = conv_params
        self.rnn_params = rnn_params
        self.deconv_params = deconv_params
        self.data_format = data_format

    def __call__(self, inputs, training, name="attention_network", reuse=None):

        with tf.variable_scope(name, reuse=reuse):

            for i, conv_param in enumerate(self.conv_params):

                with tf.variable_scope("conv_block_{}".format(i)):

                    inputs = compose(
                        lambda inputs: tf.layers.conv2d(
                            inputs=inputs,
                            filters=conv_param.filters,
                            kernel_size=conv_param.kernel_size,
                            strides=conv_param.strides,
                            padding="same",
                            data_format=self.data_format,
                            use_bias=False,
                            kernel_initializer=tf.variance_scaling_initializer(
                                scale=2.0,
                                mode="fan_in",
                                distribution="untruncated_normal"
                            ),
                            name="conv2d",
                            reuse=None
                        ),
                        lambda inputs: ops.batch_normalization(
                            inputs=inputs,
                            data_format=self.data_format,
                            training=training,
                            name="batch_normalization",
                            reuse=None
                        ),
                        lambda inputs: tf.nn.relu(inputs)
                    )(inputs)

            shape = inputs.get_shape().as_list()

            inputs = map_innermost_element(
                function=lambda inputs: tf.layers.flatten(inputs),
                sequence=inputs
            )

            feature_maps = inputs

            for i, rnn_param in enumerate(self.rnn_params[:1]):

                with tf.variable_scope("rnn_block_{}".format(i)):

                    irnn_cell = rnn_cell.IRNNCell(
                        input_units=feature_maps.shape[-1],
                        hidden_units=rnn_param.hidden_units,
                        output_units=rnn_param.output_units,
                        kernel_initializer=tf.variance_scaling_initializer(
                            scale=2.0,
                            mode="fan_in",
                            distribution="untruncated_normal"
                        ),
                        bias_initializer=tf.zeros_initializer()
                    )

                    inputs = map_innermost_element(
                        function=lambda inputs: static_rnn(
                            cell=irnn_cell,
                            inputs=[feature_maps] * rnn_param.sequence_length,
                            initial_state=irnn_cell.zero_state(
                                batch_size=tf.shape(inputs)[0],
                                dtype=tf.float32
                            )
                        ),
                        sequence=inputs
                    )

            for i, rnn_param in enumerate(self.rnn_params[1:], i + 1):

                with tf.variable_scope("rnn_block_{}".format(i)):

                    irnn_cell = rnn_cell.IRNNCell(
                        input_units=feature_maps.shape[-1],
                        hidden_units=rnn_param.hidden_units,
                        output_units=rnn_param.output_units,
                        kernel_initializer=tf.variance_scaling_initializer(
                            scale=2.0,
                            mode="fan_in",
                            distribution="untruncated_normal"
                        ),
                        bias_initializer=tf.zeros_initializer()
                    )

                    inputs = map_innermost_element(
                        function=lambda inputs: static_rnn(
                            cell=irnn_cell,
                            inputs=[feature_maps] * rnn_param.sequence_length,
                            initial_state=inputs
                        ),
                        sequence=inputs
                    )

            inputs = map_innermost_element(
                function=lambda inputs: tf.reshape(inputs, [-1] + shape[1:]),
                sequence=inputs
            )

            for i, deconv_param in enumerate(self.deconv_params[:-1]):

                with tf.variable_scope("deconv_block_{}".format(i)):

                    inputs = map_innermost_element(
                        function=compose(
                            lambda inputs: tf.layers.conv2d_transpose(
                                inputs=inputs,
                                filters=deconv_param.filters,
                                kernel_size=deconv_param.kernel_size,
                                strides=deconv_param.strides,
                                padding="same",
                                data_format=self.data_format,
                                use_bias=False,
                                kernel_initializer=tf.variance_scaling_initializer(
                                    scale=2.0,
                                    mode="fan_in",
                                    distribution="untruncated_normal"
                                ),
                                name="deconv2d",
                                reuse=tf.AUTO_REUSE
                            ),
                            lambda inputs: ops.batch_normalization(
                                inputs=inputs,
                                data_format=self.data_format,
                                training=training,
                                name="batch_normalization",
                                reuse=tf.AUTO_REUSE
                            ),
                            lambda inputs: tf.nn.relu(inputs)
                        ),
                        sequence=inputs
                    )

            for i, deconv_param in enumerate(self.deconv_params[-1:], i + 1):

                with tf.variable_scope("deconv_block_{}".format(i)):

                    inputs = map_innermost_element(
                        function=compose(
                            lambda inputs: tf.layers.conv2d_transpose(
                                inputs=inputs,
                                filters=deconv_param.filters,
                                kernel_size=deconv_param.kernel_size,
                                strides=deconv_param.strides,
                                padding="same",
                                data_format=self.data_format,
                                use_bias=False,
                                kernel_initializer=tf.variance_scaling_initializer(
                                    scale=1.0,
                                    mode="fan_avg",
                                    distribution="untruncated_normal"
                                ),
                                name="deconv2d",
                                reuse=tf.AUTO_REUSE
                            ),
                            lambda inputs: ops.batch_normalization(
                                inputs=inputs,
                                data_format=self.data_format,
                                training=training,
                                name="batch_normalization",
                                reuse=tf.AUTO_REUSE
                            ),
                            lambda inputs: tf.nn.sigmoid(inputs)
                        ),
                        sequence=inputs
                    )

            return inputs
