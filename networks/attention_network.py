import tensorflow as tf
import numpy as np
import os
from . import ops
from algorithms import *
from itertools import *


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
                            kernel_initializer=tf.initializers.variance_scaling(
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

            image_shape = inputs.shape.as_list()

            inputs = tf.layers.flatten(inputs)

            for i, rnn_param in enumerate(self.rnn_params):

                with tf.variable_scope("rnn_block_{}".format(i)):

                    lstm_cell = tf.nn.rnn_cell.LSTMCell(
                        num_units=rnn_param.num_units,
                        use_peepholes=False,
                        activation=tf.nn.tanh,
                        initializer=tf.initializers.variance_scaling(
                            scale=1.0,
                            mode="fan_avg",
                            distribution="untruncated_normal"
                        )
                    )

                    inputs = map_innermost_element(
                        function=lambda indices_inputs: tf.unstack(
                            value=tf.nn.dynamic_rnn(
                                cell=lstm_cell,
                                inputs=tf.stack(
                                    values=[indices_inputs[1]] * rnn_param.sequence_length,
                                    axis=0
                                ),
                                sequence_length=None,
                                initial_state=lstm_cell.zero_state(
                                    batch_size=tf.shape(indices_inputs[1])[0],
                                    dtype=tf.float32
                                ),
                                parallel_iterations=os.cpu_count(),
                                swap_memory=True,
                                time_major=True
                            )[0],
                            axis=0
                        ),
                        sequence=enumerate_innermost_element(inputs)
                    )

            with tf.variable_scope("projection_block"):

                inputs = map_innermost_element(
                    function=lambda inputs: tf.layers.dense(
                        inputs=inputs,
                        units=np.prod(image_shape[1:]),
                        activation=tf.nn.relu,
                        kernel_initializer=tf.initializers.variance_scaling(
                            scale=2.0,
                            mode="fan_in",
                            distribution="untruncated_normal"
                        ),
                        bias_initializer=tf.zeros_initializer(),
                        name="dense",
                        reuse=tf.AUTO_REUSE
                    ),
                    sequence=inputs
                )

            inputs = map_innermost_element(
                function=lambda inputs: tf.reshape(inputs, [-1] + image_shape[1:]),
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
                                kernel_initializer=tf.initializers.variance_scaling(
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
                                kernel_initializer=tf.initializers.variance_scaling(
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
