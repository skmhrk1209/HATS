import tensorflow as tf
import numpy as np
from . import ops
from algorithms import *
from itertools import *


class HTN(object):

    def __init__(self, conv_params, rnn_params, data_format):

        self.conv_params = conv_params
        self.rnn_params = rnn_params
        self.data_format = data_format

    def __call__(self, inputs, training, name="htn", reuse=None):

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
                    )(inputs)

            for i, rnn_param in enumerate(self.rnn_params[:-1]):

                with tf.variable_scope("rnn_block_{}".format(i)):

                    multi_lstm_cell = tf.nn.rnn_cell.MultiRNNCell([
                        tf.nn.rnn_cell.LSTMCell(
                            num_units=num_units,
                            use_peepholes=True,
                            activation=tf.nn.tanh,
                            initializer=tf.variance_scaling_initializer(
                                scale=1.0,
                                mode="fan_avg",
                                distribution="normal"
                            )
                        ) for num_units in rnn_param.num_units
                    ])

                    inputs = map_innermost_element(
                        function=lambda inputs: list(zip(
                            [inputs] * rnn_param.sequence_length,
                            tf.nn.static_rnn(
                                cell=multi_lstm_cell,
                                inputs=[tf.layers.flatten(inputs)] * rnn_param.sequence_length,
                                initial_state=multi_lstm_cell.zero_state(
                                    batch_size=tf.shape(inputs)[0],
                                    dtype=tf.float32
                                )
                            )[0]
                        )),
                        sequence=inputs
                    )

                    inputs = map_innermost_element(
                        function=lambda inputs_theta: ops.transformer(
                            U=inputs_theta[0],
                            params=inputs_theta[1],
                            out_size=rnn_param.out_size
                        ),
                        sequence=inputs
                    )

            for i, rnn_param in enumerate(self.rnn_params[-1:], i + 1):

                with tf.variable_scope("rnn_block_{}".format(i)):

                    multi_lstm_cell = tf.nn.rnn_cell.MultiRNNCell([
                        tf.nn.rnn_cell.LSTMCell(
                            num_units=num_units,
                            use_peepholes=True,
                            activation=tf.nn.tanh,
                            initializer=tf.variance_scaling_initializer(
                                scale=1.0,
                                mode="fan_avg",
                                distribution="normal"
                            )
                        ) for num_units in rnn_param.num_units
                    ])

                    inputs = map_innermost_element(
                        function=lambda inputs: tf.nn.static_rnn(
                            cell=multi_lstm_cell,
                            inputs=[tf.layers.flatten(inputs)] * rnn_param.sequence_length,
                            initial_state=multi_lstm_cell.zero_state(
                                batch_size=tf.shape(inputs)[0],
                                dtype=tf.float32
                            )
                        )[0],
                        sequence=inputs
                    )

            return inputs
