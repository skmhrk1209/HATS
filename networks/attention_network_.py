import tensorflow as tf
import numpy as np
from . import ops
from algorithms import *
from itertools import *


class AttentionNetwork(object):

    def __init__(self, conv_params, rnn_params, deconv_params, data_format,
                 pretrained_model_dir=None, pretrained_model_scope=None):

        self.conv_params = conv_params
        self.rnn_params = rnn_params
        self.deconv_params = deconv_params
        self.data_format = data_format
        self.pretrained_model_dir = pretrained_model_dir
        self.pretrained_model_scope = pretrained_model_scope

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

                    rnn_cell = tf.nn.rnn_cell.BasicRNNCell(
                        num_units=rnn_param.num_units,
                        activation=tf.nn.tanh
                    )

                    inputs = map_innermost_element(
                        function=lambda inputs: static_rnn(
                            cell=rnn_cell,
                            inputs=[references] * rnn_param.sequence_length,
                            initial_state=rnn_cell.zero_state(
                                batch_size=tf.shape(inputs)[0],
                                dtype=tf.float32
                            )
                        ),
                        sequence=inputs
                    )

            for i, rnn_param in enumerate(self.rnn_params[1:], i + 1):

                with tf.variable_scope("rnn_block_{}".format(i)):

                    rnn_cell = tf.nn.rnn_cell.BasicRNNCell(
                        num_units=rnn_param.num_units,
                        activation=tf.nn.tanh
                    )

                    inputs = map_innermost_element(
                        function=lambda inputs: static_rnn(
                            cell=rnn_cell,
                            inputs=[references] * rnn_param.sequence_length,
                            initial_state=tf.layers.dense(
                                inputs=inputs,
                                units=rnn_param.num_units,
                                activation=tf.nn.tanh,
                                name="dense",
                                reuse=tf.AUTO_REUSE
                            )
                        ),
                        sequence=inputs
                    )

            with tf.variable_scope("projection_block"):

                inputs = map_innermost_element(
                    function=lambda inputs: tf.layers.dense(
                        inputs=inputs,
                        units=np.prod(shape[1:]),
                        activation=tf.nn.tanh,
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

            if self.pretrained_model_dir:

                tf.train.init_from_checkpoint(
                    ckpt_dir_or_file=self.pretrained_model_dir,
                    assignment_map={"{}/".format(self.pretrained_model_scope): "{}/".format(name)}
                )

            return inputs