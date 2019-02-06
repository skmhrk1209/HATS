import tensorflow as tf
import numpy as np
from . import ops


class PyramidResNet(object):

    def __init__(self, conv_param, pool_param, residual_params, num_groups,
                 data_format, pretrained_model_dir=None, pretrained_model_scope=None):

        self.conv_param = conv_param
        self.pool_param = pool_param
        self.residual_params = residual_params
        self.num_groups = num_groups
        self.data_format = data_format
        self.pretrained_model_dir = pretrained_model_dir
        self.pretrained_model_scope = pretrained_model_scope

    def __call__(self, inputs, training, name="pyramid_resnet", reuse=None):

        with tf.variable_scope(name, reuse=reuse):

            if self.conv_param:

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
                        distribution="normal"
                    )
                )

                inputs = ops.group_normalization(
                    inputs=inputs,
                    num_groups=self.num_groups,
                    data_format=self.data_format
                )

                inputs = tf.nn.relu(inputs)

            if self.pool_param:

                inputs = tf.layers.max_pooling2d(
                    inputs=inputs,
                    pool_size=self.pool_param.pool_size,
                    strides=self.pool_param.strides,
                    padding="same",
                    data_format=self.data_format
                )

            feature_maps_list = []

            for i, residual_param in enumerate(self.residual_params):

                for j in range(residual_param.blocks)[:1]:

                    inputs = self.residual_block(
                        inputs=inputs,
                        filters=residual_param.filters,
                        strides=residual_param.strides,
                        projection_shortcut=True,
                        num_groups=self.num_groups,
                        data_format=self.data_format,
                        training=training,
                        name="residual_block_{}_{}".format(i, j)
                    )

                for j in range(residual_param.blocks)[1:]:

                    inputs = self.residual_block(
                        inputs=inputs,
                        filters=residual_param.filters,
                        strides=[1, 1],
                        projection_shortcut=False,
                        num_groups=self.num_groups,
                        data_format=self.data_format,
                        training=training,
                        name="residual_block_{}_{}".format(i, j)
                    )

                feature_maps_list.append(inputs)

            inputs = feature_maps_list.pop()

            while feature_maps_list:

                feature_maps = feature_maps_list.pop()

                shape = feature_maps.get_shape().as_list()

                inputs = ops.bilinear_upsampling(
                    inputs=inputs,
                    size=shape[2:] if self.data_format == "channels_first" else shape[1:-1],
                    align_corners=True,
                    data_format=self.data_format
                )

                shape = inputs.get_shape().as_list()

                feature_maps = tf.layers.conv2d(
                    inputs=feature_maps,
                    filters=shape[1] if self.data_format == "channels_first" else shape[-1],
                    kernel_size=[1, 1],
                    strides=[1, 1],
                    padding="same",
                    data_format=self.data_format,
                    use_bias=False,
                    kernel_initializer=tf.variance_scaling_initializer(
                        scale=2.0,
                        mode="fan_in",
                        distribution="normal"
                    )
                )

                feature_maps = ops.group_normalization(
                    inputs=feature_maps,
                    num_groups=self.num_groups,
                    data_format=self.data_format
                )

                feature_maps = tf.nn.relu(feature_maps)

                inputs += feature_maps

        if self.pretrained_model_dir:

            tf.train.init_from_checkpoint(
                ckpt_dir_or_file=self.pretrained_model_dir,
                assignment_map={"{}/".format(self.pretrained_model_scope): "{}/".format(name)}
            )

        return inputs

    def residual_block(self, inputs, filters, strides, projection_shortcut, num_groups, data_format, training, name="residual_block", reuse=None):
        """ A single block for ResNet v1, without a bottleneck.
        Convolution then batch normalization then ReLU as described by:
        Deep Residual Learning for Image Recognition
        (https://arxiv.org/pdf/1512.03385.pdf)
        by Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun, Dec 2015.
        """

        with tf.variable_scope(name, reuse=reuse):

            shortcut = inputs

            if projection_shortcut:

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
                        distribution="normal"
                    )
                )

                shortcut = ops.group_normalization(
                    inputs=shortcut,
                    num_groups=num_groups,
                    data_format=self.data_format
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
                    distribution="normal"
                )
            )

            inputs = ops.group_normalization(
                inputs=inputs,
                num_groups=num_groups,
                data_format=self.data_format
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
                    distribution="normal"
                )
            )

            inputs = ops.group_normalization(
                inputs=inputs,
                num_groups=num_groups,
                data_format=self.data_format
            )

            inputs += shortcut

            inputs = tf.nn.relu(inputs)

            return inputs
