import tensorflow as tf
import numpy as np


class HierarchicalRecurrentAttentionNetwork(object):

    def __init__(self, conv_params, deconv_params,
                 global_bottleneck_units, local_bottleneck_units,
                 sequence_length, string_length, data_format):

        self.conv_params = conv_params
        self.deconv_params = deconv_params
        self.global_bottleneck_units = global_bottleneck_units
        self.local_bottleneck_units = local_bottleneck_units
        self.sequence_length = sequence_length
        self.string_length = string_length
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
                        padding="same",
                        data_format=self.data_format,
                        use_bias=False,
                        kernel_initializer=tf.variance_scaling_initializer(
                            scale=2.0,
                            mode="fan_in",
                            distribution="normal",
                        ),
                        name="conv2d"
                    )

                    inputs = tf.layers.batch_normalization(
                        inputs=inputs,
                        axis=1 if self.data_format == "channels_first" else 3,
                        training=training,
                        fused=True,
                        name="batch_normalization"
                    )

                    inputs = tf.nn.relu(inputs)

            shape = inputs.shape.as_list()

            with tf.variable_scope("bottleneck_block"):

                inputs = tf.layers.flatten(inputs)

                inputs_sequence = [inputs] * self.sequence_length

                multi_lstm_cell = tf.nn.rnn_cell.MultiRNNCell([
                    tf.nn.rnn_cell.LSTMCell(
                        num_units=self.global_bottleneck_units,
                        use_peepholes=True,
                        initializer=tf.variance_scaling_initializer(
                            scale=2.0,
                            mode="fan_in",
                            distribution="normal",
                        )
                    ),
                    tf.nn.rnn_cell.LSTMCell(
                        num_units=np.prod(shape[1:]),
                        use_peepholes=True,
                        initializer=tf.variance_scaling_initializer(
                            scale=2.0,
                            mode="fan_in",
                            distribution="normal",
                        )
                    )
                ])

                inputs_sequence = tf.nn.static_rnn(
                    cell=multi_lstm_cell,
                    inputs=inputs_sequence,
                    initial_state=multi_lstm_cell.zero_state(
                        batch_size=tf.shape(inputs)[0],
                        dtype=tf.float32
                    ),
                    scope="global_rnn"
                )[0]

                inputs_sequence_sequence = [
                    [inputs] * self.string_length
                    for inputs in inputs_sequence
                ]

                multi_lstm_cell = tf.nn.rnn_cell.MultiRNNCell([
                    tf.nn.rnn_cell.LSTMCell(
                        num_units=self.local_bottleneck_units,
                        use_peepholes=True,
                        initializer=tf.variance_scaling_initializer(
                            scale=2.0,
                            mode="fan_in",
                            distribution="normal",
                        )
                    ),
                    tf.nn.rnn_cell.LSTMCell(
                        num_units=np.prod(shape[1:]),
                        use_peepholes=True,
                        initializer=tf.variance_scaling_initializer(
                            scale=2.0,
                            mode="fan_in",
                            distribution="normal",
                        )
                    )
                ])

                inputs_sequence_sequence = [
                    tf.nn.static_rnn(
                        cell=multi_lstm_cell,
                        inputs=inputs_sequence,
                        initial_state=multi_lstm_cell.zero_state(
                            batch_size=tf.shape(inputs)[0],
                            dtype=tf.float32
                        ),
                        scope="local_rnn"
                    )[0] for inputs_sequence in inputs_sequence_sequence
                ]

                inputs_sequence_sequence = [[
                    tf.reshape(
                        tensor=inputs,
                        shape=[-1] + shape[1:]
                    ) for inputs in inputs_sequence
                ] for inputs_sequence in inputs_sequence_sequence]

            for i, deconv_param in enumerate(self.deconv_params):

                with tf.variable_scope("deconv_block_{}".format(i)):

                    inputs_sequence_sequence = [[
                        tf.layers.conv2d_transpose(
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
                        ) for inputs in inputs_sequence
                    ] for inputs_sequence in inputs_sequence_sequence]

                    inputs_sequence_sequence = [[
                        tf.layers.batch_normalization(
                            inputs=inputs,
                            axis=1 if self.data_format == "channels_first" else 3,
                            training=training,
                            fused=True,
                            name="batch_normalization",
                            reuse=tf.AUTO_REUSE
                        ) for inputs in inputs_sequence
                    ] for inputs_sequence in inputs_sequence_sequence]

                    if i == len(self.deconv_params) - 1:

                        inputs_sequence_sequence = [[
                            tf.nn.sigmoid(inputs)
                            for inputs in inputs_sequence
                        ] for inputs_sequence in inputs_sequence_sequence]

                    else:

                        inputs_sequence_sequence = [[
                            tf.nn.relu(inputs)
                            for inputs in inputs_sequence
                        ] for inputs_sequence in inputs_sequence_sequence]

            return inputs_sequence_sequence
