import tensorflow as tf
import numpy as np
from . import ops
from algorithms import *
from itertools import *


class AttentionNetwork(object):

    def __init__(self, conv_params, deconv_params, rnn_params, data_format):

        self.conv_params = conv_params
        self.deconv_params = deconv_params
        self.rnn_params = rnn_params
        self.data_format = data_format

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
                                    distribution="normal"
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
                        ),
                        sequence=inputs
                    )

            # ==========================================================================================
            references = inputs

            def static_rnn(cell, inputs, initial_state):

                return list(accumulate([initial_state] + inputs, lambda state, inputs: cell(inputs, state)[1]))[1:]

            for i, rnn_param in enumerate(self.rnn_params[:1]):

                with tf.variable_scope("rnn_block_{}".format(i)):

                    conv_lstm_cell = tf.contrib.rnn.Conv2DLSTMCell(
                        input_shape=tf.shape(references)[1:],
                        output_channels=rnn_param.filters,
                        kernel_shape=rnn_param.kernel_size,
                        use_bias=True,
                    )

                    inputs = map_innermost_element(
                        function=lambda inputs: static_rnn(
                            cell=conv_lstm_cell,
                            inputs=[references] * rnn_param.sequence_length,
                            initial_state=conv_lstm_cell.zero_state(
                                batch_size=tf.shape(references)[0],
                                dtype=tf.float32
                            )
                        ),
                        sequence=inputs
                    )

            for i, rnn_param in enumerate(self.rnn_params[1:], i + 1):

                with tf.variable_scope("rnn_block_{}".format(i)):

                    conv_lstm_cell = tf.contrib.rnn.Conv2DLSTMCell(
                        input_shape=tf.shape(references)[1:],
                        output_channels=rnn_param.filters,
                        kernel_shape=rnn_param.kernel_size,
                        use_bias=True,
                    )

                    inputs = map_innermost_element(
                        function=lambda inputs: static_rnn(
                            cell=conv_lstm_cell,
                            inputs=[references] * rnn_param.sequence_length,
                            initial_state=tf.nn.rnn_cell.LSTMStateTuple(
                                c=tf.layers.conv2d(
                                    inputs=inputs.c,
                                    filters=rnn_param.filters,
                                    kernel_size=[1, 1],
                                    strides=[1, 1],
                                    padding="same",
                                    data_format=self.data_format,
                                    activation=None,
                                    use_bias=True,
                                    name="c_projection",
                                    reuse=tf.AUTO_REUSE
                                ),
                                h=tf.layers.dense(
                                    inputs=inputs.h,
                                    filters=rnn_param.filters,
                                    kernel_size=[1, 1],
                                    strides=[1, 1],
                                    padding="same",
                                    data_format=self.data_format,
                                    activation=tf.nn.tanh,
                                    use_bias=True,
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
                                    distribution="normal"
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
                                    distribution="normal"
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
