import tensorflow as tf
import numpy as np


class Model(object):

    def __init__(self, convolutional_network, recurrent_attention_network,
                 num_classes, num_digits, data_format, hyper_params):

        self.convolutional_network = convolutional_network
        self.recurrent_attention_network = recurrent_attention_network
        self.num_classes = num_classes
        self.num_digits = num_digits
        self.data_format = data_format
        self.hyper_params = hyper_params

    def __call__(self, features, labels, mode):

        feature_maps = self.convolutional_network(
            inputs=features["images"],
            training=mode == tf.estimator.ModeKeys.TRAIN
        )

        attention_maps_sequence = self.recurrent_attention_network(
            inputs=feature_maps,
            training=mode == tf.estimator.ModeKeys.TRAIN
        )

        merged_attention_maps_sequence = [
            tf.reduce_sum(
                input_tensor=attention_maps,
                axis=1 if self.data_format == "channels_first" else 3,
                keep_dims=True
            ) for attention_maps in attention_maps_sequence
        ]

        def flatten_images(inputs, data_format):

            input_shape = inputs.get_shape().as_list()
            output_shape = ([-1, input_shape[1], np.prod(input_shape[2:4])] if self.data_format == "channels_first" else
                            [-1, np.prod(input_shape[1:3]), input_shape[3]])

            return tf.reshape(inputs, output_shape)

        feature_vectors_sequence = [
            tf.matmul(
                a=flatten_images(feature_maps, self.data_format),
                b=flatten_images(attention_maps, self.data_format),
                transpose_a=False if self.data_format == "channels_first" else True,
                transpose_b=True if self.data_format == "channels_first" else False
            ) for attention_maps in attention_maps_sequence
        ]

        feature_vectors_sequence = [
            tf.layers.flatten(
                inputs=feature_vectors
            ) for feature_vectors in feature_vectors_sequence
        ]

        multi_logits_sequence = [[
            tf.layers.dense(
                inputs=feature_vectors,
                units=self.num_classes,
                kernel_initializer=tf.variance_scaling_initializer(
                    scale=2.0,
                    mode="fan_in",
                    distribution="normal",
                ),
                bias_initializer=tf.zeros_initializer(),
                name="logits_{}".format(i),
                reuse=tf.AUTO_REUSE
            ) for i in range(self.num_digits)
        ] for feature_vectors in feature_vectors_sequence]

        multi_labels_sequence = [
            tf.unstack(multi_labels, axis=1)
            for multi_labels in tf.unstack(labels, axis=1)
        ]

        multi_cross_entropy_loss_sequence = [[
            tf.losses.sparse_softmax_cross_entropy(
                labels=labels,
                logits=logits
            ) for labels, logits in zip(multi_labels, multi_logits)
        ] for multi_labels, multi_logits in zip(multi_labels_sequence, multi_logits_sequence)]

        cross_entropy_loss = tf.reduce_mean([
            tf.reduce_mean(multi_cross_entropy_loss)
            for multi_cross_entropy_loss in multi_cross_entropy_loss_sequence
        ])

        attention_map_loss = tf.reduce_mean([
            tf.reduce_mean(tf.reduce_sum(tf.abs(attention_maps), axis=[1, 2, 3]))
            for attention_maps in attention_maps_sequence
        ])

        total_variation_loss = tf.reduce_mean([
            tf.reduce_mean(tf.image.total_variation(attention_maps))
            for attention_maps in attention_maps_sequence
        ])

        loss = \
            cross_entropy_loss * self.hyper_params.cross_entropy_decay + \
            attention_map_loss * self.hyper_params.attention_map_decay + \
            total_variation_loss * self.hyper_params.total_variation_decay \

        multi_classes_sequence = [[
            tf.argmax(logits, axis=-1)
            for logits in multi_logits
        ] for multi_logits in multi_logits_sequence]

        accuracy = tf.metrics.accuracy(
            labels=labels,
            predictions=tf.concat([
                tf.stack(multi_classes, axis=-1)
                for multi_classes in multi_classes_sequence
            ], axis=-1)
        )

        # ==========================================================================================
        tf.summary.image("images", features["images"], max_outputs=4)
        [tf.summary.image("merged_attention_maps_sequence_{}".format(i), merged_attention_maps, max_outputs=2)
         for i, merged_attention_maps in enumerate(merged_attention_maps_sequence)]
        tf.summary.scalar("cross_entropy_loss", cross_entropy_loss)
        tf.summary.scalar("attention_map_loss", attention_map_loss)
        tf.summary.scalar("total_variation_loss", total_variation_loss)
        tf.summary.scalar("loss", loss)
        tf.summary.scalar("accuracy", accuracy[1])
        # ==========================================================================================

        if mode == tf.estimator.ModeKeys.TRAIN:

            with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):

                train_op = tf.train.AdamOptimizer().minimize(
                    loss=loss,
                    global_step=tf.train.get_global_step()
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

        if mode == tf.estimator.ModeKeys.PREDICT:

            features.update({
                "merged_attention_maps_{}".format(i): merged_attention_maps
                for i, merged_attention_maps in enumerate(merged_attention_maps_sequence)
            })

            return tf.estimator.EstimatorSpec(
                mode=mode,
                predictions=features
            )
