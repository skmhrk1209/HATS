import tensorflow as tf
import numpy as np


class RecurrentAttentionNetwork(object):

    def __init__(self, conv_params, deconv_params, bottleneck_units, sequence_length, data_format):

        self.conv_params = conv_params
        self.deconv_params = deconv_params
        self.bottleneck_units = bottleneck_units
        self.sequence_length = sequence_length
        self.data_format = data_format

    def __call__(self, inputs, training, name="attention_network", reuse=None):

        with tf.variable_scope(name, reuse=reuse):

            inputs_sequence = [inputs] * self.sequence_length

            for i, conv_param in enumerate(self.conv_params):

                with tf.variable_scope("conv_block_{}".format(i)):

                    inputs_sequence = [tf.layers.conv2d(
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
                        reuse=tf.AUTO_REUSE
                    ) for inputs in inputs_sequence]

                    inputs_sequence = [tf.layers.batch_normalization(
                        inputs=inputs,
                        axis=1 if self.data_format == "channels_first" else 3,
                        momentum=0.997,
                        epsilon=1e-5,
                        center=True,
                        scale=True,
                        training=training,
                        fused=True,
                        name="batch_normalization",
                        reuse=tf.AUTO_REUSE
                    ) for inputs in inputs_sequence]

                    inputs_sequence = [tf.nn.relu(
                        features=inputs
                    ) for inputs in inputs_sequence]

            shape = inputs_sequence[0].shape.as_list()

            with tf.variable_scope("bottleneck_block"):

                inputs_sequence = [tf.layers.flatten(
                    inputs=inputs
                ) for inputs in inputs_sequence]

                multi_lstm_cell = tf.nn.rnn_cell.MultiRNNCell([
                    tf.nn.rnn_cell.LSTMCell(
                        num_units=self.bottleneck_units,
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

                inputs_sequence, _ = tf.nn.static_rnn(
                    cell=multi_lstm_cell,
                    inputs=inputs_sequence,
                    dtype=tf.float32
                )

                inputs_sequence = [tf.reshape(
                    tensor=inputs,
                    shape=[-1] + shape[1:]
                ) for inputs in inputs_sequence]

            for i, deconv_param in enumerate(self.deconv_params):

                with tf.variable_scope("deconv_block_{}".format(i)):

                    inputs_sequence = [tf.layers.conv2d_transpose(
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
                    ) for inputs in inputs_sequence]

                    inputs_sequence = [tf.layers.batch_normalization(
                        inputs=inputs,
                        axis=1 if self.data_format == "channels_first" else 3,
                        momentum=0.997,
                        epsilon=1e-5,
                        center=True,
                        scale=True,
                        training=training,
                        fused=True,
                        name="batch_normalization",
                        reuse=tf.AUTO_REUSE
                    ) for inputs in inputs_sequence]

                    if i == len(self.deconv_params) - 1:

                        inputs_sequence = [tf.nn.sigmoid(
                            x=inputs
                        ) for inputs in inputs_sequence]

                    else:

                        inputs_sequence = [tf.nn.relu(
                            features=inputs
                        ) for inputs in inputs_sequence]

            return inputs_sequence
