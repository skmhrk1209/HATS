import tensorflow as tf
import numpy as np
from . import ops
from algorithms import *
from itertools import *


class HAN(object):

    def __init__(self, conv_params, rnn_params, deconv_params, data_format):

        self.conv_params = conv_params
        self.rnn_params = rnn_params
        self.deconv_params = deconv_params
        self.data_format = data_format

    def __call__(self, inputs, training, name="han", reuse=None):

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

            shape = inputs.get_shape().as_list()

            inputs = map_innermost_element(
                function=lambda inputs: tf.layers.flatten(inputs),
                sequence=inputs
            )

            references = inputs

            def static_rnn(cell, inputs, initial_state):

                return list(accumulate([initial_state] + inputs, lambda state, inputs: cell(inputs, state)[1]))[1:]

            for i, rnn_param in enumerate(self.rnn_params[:1]):

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
                        function=lambda inputs: static_rnn(
                            cell=multi_lstm_cell,
                            inputs=[references] * rnn_param.sequence_length,
                            initial_state=multi_lstm_cell.zero_state(
                                batch_size=tf.shape(inputs)[0],
                                dtype=tf.float32
                            )
                        ),
                        sequence=inputs
                    )

            for i, rnn_param in enumerate(self.rnn_params[1:], i + 1):

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
                        function=lambda inputs: static_rnn(
                            cell=multi_lstm_cell,
                            inputs=[references] * rnn_param.sequence_length,
                            initial_state=map_innermost_element(
                                lambda index_inputs_num_units: tf.nn.rnn_cell.LSTMStateTuple(
                                    c=tf.layers.dense(
                                        inputs=index_inputs_num_units[1][0].c,
                                        units=index_inputs_num_units[1][1],
                                        activation=None,
                                        kernel_initializer=tf.variance_scaling_initializer(
                                            scale=1.0,
                                            mode="fan_avg",
                                            distribution="normal"
                                        ),
                                        bias_initializer=tf.zeros_initializer(),
                                        name="c_projection_{}".format(index_inputs_num_units[0]),
                                        reuse=tf.AUTO_REUSE
                                    ),
                                    h=tf.layers.dense(
                                        inputs=index_inputs_num_units[1][0].h,
                                        units=index_inputs_num_units[1][1],
                                        activation=tf.nn.tanh,
                                        kernel_initializer=tf.variance_scaling_initializer(
                                            scale=1.0,
                                            mode="fan_avg",
                                            distribution="normal"
                                        ),
                                        bias_initializer=tf.zeros_initializer(),
                                        name="h_projection_{}".format(index_inputs_num_units[0]),
                                        reuse=tf.AUTO_REUSE
                                    )
                                ), list(enumerate(zip(inputs, rnn_param.num_units)))
                            )
                        ),
                        sequence=inputs
                    )

            inputs = map_innermost_element(
                function=lambda inputs: inputs[-1].h,
                sequence=inputs
            )

            with tf.variable_scope("projection_block"):

                inputs = map_innermost_element(
                    function=lambda inputs: tf.layers.dense(
                        inputs=inputs,
                        units=np.prod(shape[1:]),
                        activation=tf.nn.tanh,
                        kernel_initializer=tf.variance_scaling_initializer(
                            scale=1.0,
                            mode="fan_avg",
                            distribution="normal"
                        ),
                        bias_initializer=tf.zeros_initializer(),
                        name="dense",
                        reuse=tf.AUTO_REUSE
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
