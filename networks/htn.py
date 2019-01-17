import tensorflow as tf
import numpy as np
from . import ops
from algorithms import *
from itertools import *


class HTN(object):

    def __init__(self, rnn_params, data_format):

        self.rnn_params = rnn_params
        self.data_format = data_format

    def __call__(self, inputs, training, name="htn", reuse=None):

        for i, rnn_param in enumerate(self.rnn_params):

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
                        theta=inputs_theta[1],
                        out_size=rnn_param.out_size
                    ),
                    sequence=inputs
                )

        return inputs
