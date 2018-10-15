import tensorflow as tf
import numpy as np


class Model(object):

    def __init__(self, convolutional_network, attention_network,
                 num_classes, num_digits, data_format, hyper_params):

        self.convolutional_network = convolutional_network
        self.attention_network = attention_network
        self.num_classes = num_classes
        self.num_digits = num_digits
        self.data_format = data_format
        self.hyper_params = hyper_params

    def __call__(self, features, labels, mode):
        ''' model function for ACNN

            features:   batch of features from input_fn
            labels:     batch of labels from input_fn
            mode:       enum { TRAIN, EVAL, PREDICT }
        '''

        tf.summary.image(
            name="features",
            tensor=features,
            max_outputs=10
        )

        feature_maps = self.convolutional_network(
            inputs=features,
            training=mode == tf.estimator.ModeKeys.TRAIN
        )

        attention_maps = self.attention_network(
            inputs=feature_maps,
            training=mode == tf.estimator.ModeKeys.TRAIN
        )

        reduced_attention_maps = tf.reduce_sum(
            input_tensor=attention_maps,
            axis=1 if self.data_format == "channels_first" else 3,
            keep_dims=True
        )

        tf.summary.image(
            name="reduced_attention_maps",
            tensor=reduced_attention_maps,
            max_outputs=10
        )

        def flatten_images(inputs, data_format):

            input_shape = inputs.get_shape().as_list()
            output_shape = ([-1, input_shape[1], np.prod(input_shape[2:4])] if self.data_format == "channels_first" else
                            [-1, np.prod(input_shape[1:3]), input_shape[3]])

            return tf.reshape(inputs, output_shape)

        feature_vectors = tf.matmul(
            a=flatten_images(feature_maps, self.data_format),
            b=flatten_images(attention_maps, self.data_format),
            transpose_a=False if self.data_format == "channels_first" else True,
            transpose_b=True if self.data_format == "channels_first" else False
        )

        feature_vectors = tf.layers.flatten(feature_vectors)

        multi_logits = tf.stack([
            tf.layers.dense(
                inputs=feature_vectors,
                units=self.num_classes,
                kernel_initializer=tf.variance_scaling_initializer(
                    scale=2.0,
                    mode="fan_in",
                    distribution="normal",
                ),
                bias_initializer=tf.zeros_initializer()
            ) for i in range(self.num_digits)
        ], axis=1)

        softmax = tf.nn.softmax(multi_logits, dim=-1, name="softmax")
        classes = tf.argmax(multi_logits, axis=-1, name="classes")

        if mode == tf.estimator.ModeKeys.PREDICT:

            return tf.estimator.EstimatorSpec(
                mode=mode,
                predictions=dict(
                    features=features,
                    feature_maps=feature_maps,
                    attention_maps=attention_maps,
                    reduced_attention_maps=reduced_attention_maps,
                    feature_vectors=feature_vectors,
                    softmax=softmax,
                    classes=classes
                )
            )

        cross_entropy_loss = tf.reduce_mean([
            tf.losses.sparse_softmax_cross_entropy(
                labels=labels[:, i],
                logits=multi_logits[:, i, :]
            ) for i in range(self.num_digits)
        ])

        attention_map_loss = tf.reduce_mean(tf.reduce_sum(tf.abs(attention_maps), [1, 2, 3]))
        total_variation_loss = tf.reduce_mean(tf.image.total_variation(attention_maps))

        loss = cross_entropy_loss + \
            attention_map_loss * self.hyper_params.attention_map_decay + \
            total_variation_loss * self.hyper_params.total_variation_decay \

        tf.summary.scalar("cross_entropy_loss", cross_entropy_loss)
        tf.summary.scalar("attention_map_loss", attention_map_loss)
        tf.summary.scalar("total_variation_loss", total_variation_loss)
        tf.summary.scalar("loss", loss)

        accuracy = tf.metrics.accuracy(
            labels=labels,
            predictions=classes
        )

        tf.summary.scalar("accuracy", accuracy[1])

        if mode == tf.estimator.ModeKeys.TRAIN:

            with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):

                global_step = tf.train.get_or_create_global_step()

                optimizer = tf.train.AdamOptimizer()

                train_op = optimizer.minimize(
                    loss=loss,
                    global_step=global_step
                )

            return tf.estimator.EstimatorSpec(
                mode=mode,
                loss=loss,
                train_op=train_op
            )

        if mode == tf.estimator.ModeKeys.EVAL:

            return tf.estimator.EstimatorSpec(
                mode=mode,
                loss=loss,
                eval_metric_ops={"accuracy": accuracy}
            )
