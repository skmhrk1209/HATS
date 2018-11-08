import tensorflow as tf
import numpy as np


class Model(object):

    def __init__(self, convolutional_network, attention_network,
                 string_length, num_classes, data_format, hyper_params):

        self.convolutional_network = convolutional_network
        self.attention_network = attention_network
        self.string_length = string_length
        self.num_classes = num_classes
        self.data_format = data_format
        self.hyper_params = hyper_params

    def __call__(self, features, labels, mode):

        images = features["image"]

        attention_maps_sequence = self.attention_network(
            inputs=images,
            training=mode == tf.estimator.ModeKeys.TRAIN,
            name="attention_network"
        )

        images_sequence = [
            images * attention_maps
            for attention_maps in attention_maps_sequence
        ]

        feature_maps_sequence = [
            self.convolutional_network(
                inputs=images,
                training=mode == tf.estimator.ModeKeys.TRAIN,
                name="convolutional_network",
                reuse=tf.AUTO_REUSE
            ) for images in images_sequence
        ]

        feature_vectors_sequence = [
            tf.reduce_mean(
                input_tensor=feature_maps,
                axis=[2, 3] if self.data_format == "channels_first" else [1, 2]
            ) for feature_maps in feature_maps_sequence
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
            ) for i in range(self.string_length)
        ] for feature_vectors in feature_vectors_sequence]

        if mode == tf.estimator.ModeKeys.PREDICT:

            return tf.estimator.EstimatorSpec(
                mode=mode,
                predictions={
                    *{"images": images},
                    *{"attention_maps_{}".format(i): attention_maps
                      for i, attention_maps in enumerate(attention_maps_sequence)}
                }
            )

        multi_labels_sequence = [
            tf.unstack(multi_labels, axis=1)
            for multi_labels in tf.unstack(labels, axis=1)
        ]

        cross_entropy_loss = tf.reduce_mean([[
            tf.losses.sparse_softmax_cross_entropy(
                labels=labels,
                logits=logits
            ) for labels, logits in zip(multi_labels, multi_logits)
        ] for multi_labels, multi_logits in zip(multi_labels_sequence, multi_logits_sequence)])

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
            total_variation_loss * self.hyper_params.total_variation_decay

        multi_classes_sequence = [[
            tf.argmax(logits, axis=-1)
            for logits in multi_logits
        ] for multi_logits in multi_logits_sequence]

        accuracy = tf.metrics.accuracy(
            labels=multi_labels_sequence,
            predictions=multi_classes_sequence
        )

        non_streaming_accuracy = tf.reduce_mean(tf.cast(multi_classes_sequence == multi_labels_sequence, tf.float32))
        tf.identity(non_streaming_accuracy, name="non_streaming_accuracy")

        # ==========================================================================================
        tf.summary.image("images", images, max_outputs=4)
        [tf.summary.image("attention_maps_{}".format(i), attention_maps, max_outputs=4)
         for i, attention_maps in enumerate(attention_maps_sequence)]
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
