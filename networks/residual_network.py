import tensorflow as tf
import numpy as np


class ResidualNetwork(object):

    def __init__(self, conv_param, pool_param, residual_params, num_classes, channels_first):

        self.conv_param = conv_param
        self.pool_param = pool_param
        self.residual_params = residual_params
        self.num_classes = num_classes
        self.channels_first = channels_first
        self.data_format = "channels_first" if channels_first else "channels_last"

    def __call__(self, inputs, training, name="residual_network", reuse=None):

        with tf.variable_scope(name, reuse=reuse):

            inputs = tf.layers.conv2d(
                inputs=inputs,
                filters=self.conv_param.filters,
                kernel_size=self.conv_param.kernel_size,
                strides=self.conv_param.strides,
                padding="same",
                data_format=self.data_format,
                use_bias=False,
                kernel_initializer=tf.variance_scaling_initializer(
                    scale=2.0,
                    mode="fan_in",
                    distribution="normal",
                )
            )

            if self.pool_param:

                inputs = tf.layers.max_pooling2d(
                    inputs=inputs,
                    pool_size=self.pool_param.pool_size,
                    strides=self.pool_param.strides,
                    padding="same",
                    data_format=self.data_format
                )

            for i, residual_param in enumerate(self.residual_params):

                for j in range(residual_param.blocks)[:1]:

                    inputs = self.residual_block(
                        inputs=inputs,
                        filters=residual_param.filters,
                        strides=residual_param.strides,
                        data_format=self.data_format,
                        training=training,
                        name="residual_block_{}_{}".format(i, j)
                    )

                for j in range(residual_param.blocks)[1:]:

                    inputs = self.residual_block(
                        inputs=inputs,
                        filters=residual_param.filters,
                        strides=[1, 1],
                        data_format=self.data_format,
                        training=training,
                        name="residual_block_{}_{}".format(i, j)
                    )

            inputs = tf.layers.batch_normalization(
                inputs=inputs,
                axis=1 if self.channels_first else 3,
                training=training,
                fused=True
            )

            inputs = tf.nn.relu(inputs)

            if not self.num_classes:

                return inputs

            inputs = tf.reduce_mean(
                input_tensor=inputs,
                axis=[2, 3] if self.channels_first else [1, 2]
            )

            inputs = tf.layers.dense(
                inputs=inputs,
                units=self.num_classes,
                kernel_initializer=tf.variance_scaling_initializer(
                    scale=2.0,
                    mode="fan_in",
                    distribution="normal",
                ),
                bias_initializer=tf.zeros_initializer()
            )

            return inputs

    def residual_block(self, inputs, filters, strides, data_format, training, name="residual_block", reuse=None):
        ''' preactivation building residual block
            normalization then activation then convolution as described by:
            [Identity Mappings in Deep Residual Networks]
            (https://arxiv.org/pdf/1603.05027.pdf)
        '''

        with tf.variable_scope(name, reuse=reuse):

            inputs = tf.layers.batch_normalization(
                inputs=inputs,
                axis=1 if self.channels_first else 3,
                training=training,
                fused=True
            )

            inputs = tf.nn.relu(inputs)

            shortcut = tf.layers.conv2d(
                inputs=inputs,
                filters=filters,
                kernel_size=[1, 1],
                strides=strides,
                padding="same",
                data_format=data_format,
                use_bias=False,
                kernel_initializer=tf.variance_scaling_initializer(
                    scale=2.0,
                    mode="fan_in",
                    distribution="normal",
                )
            )

            inputs = tf.layers.conv2d(
                inputs=inputs,
                filters=filters,
                kernel_size=[3, 3],
                strides=strides,
                padding="same",
                data_format=data_format,
                use_bias=False,
                kernel_initializer=tf.variance_scaling_initializer(
                    scale=2.0,
                    mode="fan_in",
                    distribution="normal",
                )
            )

            inputs = tf.layers.batch_normalization(
                inputs=inputs,
                axis=1 if self.channels_first else 3,
                training=training,
                fused=True
            )

            inputs = tf.nn.relu(inputs)

            inputs = tf.layers.conv2d(
                inputs=inputs,
                filters=filters,
                kernel_size=[3, 3],
                strides=[1, 1],
                padding="same",
                data_format=data_format,
                use_bias=False,
                kernel_initializer=tf.variance_scaling_initializer(
                    scale=2.0,
                    mode="fan_in",
                    distribution="normal",
                )
            )

            inputs += shortcut

            return inputs
