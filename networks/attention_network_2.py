import tensorflow as tf
import numpy as np
import os
from itertools import *
from algorithms import *


class AttentionNetwork(object):

    def __init__(self, conv_params, deconv_params, rnn_params, channels_first):

        self.conv_params = conv_params
        self.deconv_params = deconv_params
        self.rnn_params = rnn_params
        self.channels_first = channels_first
        self.data_format = "channels_first" if channels_first else "channels_last"

    def __call__(self, inputs, training, name="attention_network", reuse=None):

        with tf.variable_scope(name, reuse=reuse):

            for i, conv_param in enumerate(self.conv_params):

                with tf.variable_scope("conv_block_{}".format(i)):

                    inputs = map_innermost_element(
                        function=compose(
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
                                    distribution="normal",
                                ),
                                name="conv2d",
                                reuse=None
                            ),
                            lambda inputs: tf.layers.batch_normalization(
                                inputs=inputs,
                                axis=1 if self.channels_first else 3,
                                training=training,
                                fused=True,
                                name="batch_normalization",
                                reuse=None
                            ),
                            lambda inputs: tf.nn.relu(inputs)
                        ),
                        sequence=inputs
                    )

            shape = inputs.get_shape().as_list()

            inputs = map_innermost_element(
                function=lambda inputs: tf.layers.flatten(inputs),
                sequence=inputs
            )

            # ==========================================================================================
            references = inputs

            def static_rnn(cell, inputs, initial_state):

                return list(accumulate([initial_state] + inputs, lambda state, inputs: cell(inputs, state)[1]))[1:]

            for i, rnn_param in enumerate(self.rnn_params[:1]):

                with tf.variable_scope("rnn_block_{}".format(i)):

                    lstm_cell = tf.nn.rnn_cell.LSTMCell(
                        num_units=rnn_param.num_units,
                        use_peepholes=True,
                        activation=tf.nn.tanh,
                        initializer=tf.variance_scaling_initializer(
                            scale=1.0,
                            mode="fan_avg",
                            distribution="normal",
                        )
                    )

                    print(lstm_cell.zero_state)

                    inputs = map_innermost_element(
                        function=lambda inputs: static_rnn(
                            cell=lstm_cell,
                            inputs=[references] * rnn_param.sequence_length,
                            initial_state=lstm_cell.zero_state(
                                batch_size=tf.shape(inputs)[0],
                                dtype=tf.float32
                            )
                        ),
                        sequence=inputs
                    )

            for i, rnn_param in enumerate(self.rnn_params[1:-1], i + 1):

                with tf.variable_scope("rnn_block_{}".format(i)):

                    lstm_cell = tf.nn.rnn_cell.LSTMCell(
                        num_units=rnn_param.num_units,
                        use_peepholes=True,
                        activation=tf.nn.tanh,
                        initializer=tf.variance_scaling_initializer(
                            scale=1.0,
                            mode="fan_avg",
                            distribution="normal",
                        )
                    )

                    inputs = map_innermost_element(
                        function=lambda inputs: static_rnn(
                            cell=lstm_cell,
                            inputs=[references] * rnn_param.sequence_length,
                            initial_state=tf.nn.rnn_cell.LSTMStateTuple(
                                c=tf.layers.dense(
                                    inputs=inputs.c,
                                    units=rnn_param.num_units,
                                    activation=tf.nn.tanh,
                                    kernel_initializer=tf.variance_scaling_initializer(
                                        scale=1.0,
                                        mode="fan_avg",
                                        distribution="normal",
                                    ),
                                    bias_initializer=tf.zeros_initializer(),
                                    name="c_projection",
                                    reuse=tf.AUTO_REUSE
                                ),
                                h=tf.layers.dense(
                                    inputs=inputs.h,
                                    units=rnn_param.num_units,
                                    activation=tf.nn.tanh,
                                    kernel_initializer=tf.variance_scaling_initializer(
                                        scale=1.0,
                                        mode="fan_avg",
                                        distribution="normal",
                                    ),
                                    bias_initializer=tf.zeros_initializer(),
                                    name="h_projection",
                                    reuse=tf.AUTO_REUSE
                                )
                            )
                        ),
                        sequence=inputs
                    )

            for i, rnn_param in enumerate(self.rnn_params[-1:], i + 1):

                with tf.variable_scope("rnn_block_{}".format(i)):

                    lstm_cell = tf.nn.rnn_cell.LSTMCell(
                        num_units=rnn_param.num_units,
                        use_peepholes=True,
                        activation=tf.nn.tanh,
                        initializer=tf.variance_scaling_initializer(
                            scale=1.0,
                            mode="fan_avg",
                            distribution="normal",
                        ),
                        num_proj=np.prod(shape[1:])
                    )

                    inputs = map_innermost_element(
                        function=lambda inputs: static_rnn(
                            cell=lstm_cell,
                            inputs=[references] * rnn_param.sequence_length,
                            initial_state=tf.nn.rnn_cell.LSTMStateTuple(
                                c=tf.layers.dense(
                                    inputs=inputs.c,
                                    units=rnn_param.num_units,
                                    activation=tf.nn.tanh,
                                    kernel_initializer=tf.variance_scaling_initializer(
                                        scale=1.0,
                                        mode="fan_avg",
                                        distribution="normal",
                                    ),
                                    bias_initializer=tf.zeros_initializer(),
                                    name="c_projection",
                                    reuse=tf.AUTO_REUSE
                                ),
                                h=tf.layers.dense(
                                    inputs=inputs.h,
                                    units=rnn_param.num_units,
                                    activation=tf.nn.tanh,
                                    kernel_initializer=tf.variance_scaling_initializer(
                                        scale=1.0,
                                        mode="fan_avg",
                                        distribution="normal",
                                    ),
                                    bias_initializer=tf.zeros_initializer(),
                                    name="h_projection",
                                    reuse=tf.AUTO_REUSE
                                )
                            )
                        ),
                        sequence=inputs
                    )

            inputs = map_innermost_element(
                function=lambda inputs: inputs.h,
                sequence=inputs
            )

            # ==========================================================================================

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
                                    distribution="normal",
                                ),
                                name="deconv2d",
                                reuse=tf.AUTO_REUSE
                            ),
                            lambda inputs: tf.layers.batch_normalization(
                                inputs=inputs,
                                axis=1 if self.channels_first else 3,
                                training=training,
                                fused=True,
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
                                    distribution="normal",
                                ),
                                name="deconv2d",
                                reuse=tf.AUTO_REUSE
                            ),
                            lambda inputs: tf.layers.batch_normalization(
                                inputs=inputs,
                                axis=1 if self.channels_first else 3,
                                training=training,
                                fused=True,
                                name="batch_normalization",
                                reuse=tf.AUTO_REUSE
                            ),
                            lambda inputs: tf.nn.sigmoid(inputs)
                        ),
                        sequence=inputs
                    )

            return inputs
