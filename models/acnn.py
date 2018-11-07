import tensorflow as tf
import numpy as np


class Model(object):

    def __init__(self, convolutional_network, attention_network,
                 num_classes, data_format, hyper_params):

        self.convolutional_network = convolutional_network
        self.attention_network = attention_network
        self.num_classes = num_classes
        self.data_format = data_format
        self.hyper_params = hyper_params

    def __call__(self, features, labels, mode):

        tf.summary.image("images", features["images"], max_outputs=2)

        feature_maps = self.convolutional_network(
            inputs=features["images"],
            training=mode == tf.estimator.ModeKeys.TRAIN
        )

        attention_maps_sequence_sequence = self.attention_network(
            inputs=feature_maps,
            training=mode == tf.estimator.ModeKeys.TRAIN
        )

        merged_attention_maps_sequence_sequence = [[
            tf.reduce_sum(
                input_tensor=attention_maps,
                axis=1 if self.data_format == "channels_first" else 3,
                keep_dims=True
            ) for attention_maps in attention_maps_sequence
        ] for attention_maps_sequence in attention_maps_sequence_sequence]

        for i, merged_attention_maps_sequence in enumerate(merged_attention_maps_sequence_sequence):
            for j, merged_attention_maps in enumerate(merged_attention_maps_sequence):
                tf.summary.image("merged_attention_maps_sequence_sequence_{}_{}".format(i, j), merged_attention_maps, max_outputs=2)

        def flatten_images(inputs, data_format):

            input_shape = inputs.get_shape().as_list()
            output_shape = ([-1, input_shape[1], np.prod(input_shape[2:4])] if self.data_format == "channels_first" else
                            [-1, np.prod(input_shape[1:3]), input_shape[3]])

            return tf.reshape(inputs, output_shape)

        feature_vectors_sequence_sequence = [[
            tf.matmul(
                a=flatten_images(feature_maps, self.data_format),
                b=flatten_images(attention_maps, self.data_format),
                transpose_a=False if self.data_format == "channels_first" else True,
                transpose_b=True if self.data_format == "channels_first" else False
            ) for attention_maps in attention_maps_sequence
        ] for attention_maps_sequence in attention_maps_sequence_sequence]

        feature_vectors_sequence_sequence = [[
            tf.layers.flatten(
                inputs=feature_vectors
            ) for feature_vectors in feature_vectors_sequence
        ] for feature_vectors_sequence in feature_vectors_sequence_sequence]

        logits_sequence_sequence = [[
            tf.layers.dense(
                inputs=feature_vectors,
                units=self.num_classes,
                kernel_initializer=tf.variance_scaling_initializer(
                    scale=2.0,
                    mode="fan_in",
                    distribution="normal",
                ),
                bias_initializer=tf.zeros_initializer(),
                name="logits",
                reuse=tf.AUTO_REUSE
            ) for feature_vectors in feature_vectors_sequence
        ] for feature_vectors_sequence in feature_vectors_sequence_sequence]

        classes_sequence_sequence = [[
            tf.argmax(
                input=logits,
                axis=-1,
                name="classes"
            ) for logits in logits_sequence
        ] for logits_sequence in logits_sequence_sequence]

        if mode == tf.estimator.ModeKeys.PREDICT:

            features.update({
                "merged_attention_maps_sequence_sequence": tf.stack([
                    tf.stack(merged_attention_maps_sequence, axis=1)
                    for merged_attention_maps_sequence in merged_attention_maps_sequence_sequence
                ], axis=1),
                "classes_sequence_sequence": tf.stack([
                    tf.stack(classes_sequence, axis=1)
                    for classes_sequence in classes_sequence_sequence
                ], axis=1)
            })

            return tf.estimator.EstimatorSpec(
                mode=mode,
                predictions=features
            )

        labels_sequence_sequence = [
            tf.unstack(multi_labels, axis=1)
            for multi_labels in tf.unstack(labels, axis=1)
        ]

        cross_entropy_loss_sequence_sequence = [[
            tf.losses.sparse_softmax_cross_entropy(
                labels=labels,
                logits=logits
            ) for labels, logits in zip(labels_sequence, logits_sequence)
        ] for labels_sequence, logits_sequence in zip(labels_sequence_sequence, logits_sequence_sequence)]

        for i, cross_entropy_loss_sequence in enumerate(cross_entropy_loss_sequence_sequence):
            for j, cross_entropy_loss in enumerate(cross_entropy_loss_sequence):
                tf.summary.scalar("cross_entropy_loss_sequence_sequence_{}_{}".format(i, j), cross_entropy_loss)

        cross_entropy_loss_sequence = [
            tf.reduce_mean(cross_entropy_loss_sequence)
            for cross_entropy_loss_sequence in cross_entropy_loss_sequence_sequence
        ]

        for i, cross_entropy_loss in enumerate(cross_entropy_loss_sequence):
            tf.summary.scalar("cross_entropy_loss_sequence_{}".format(i), cross_entropy_loss)

        cross_entropy_loss = tf.reduce_mean(cross_entropy_loss_sequence)

        tf.summary.scalar("cross_entropy_loss", cross_entropy_loss)

        attention_map_loss_sequence_sequence = [[
            tf.reduce_mean(tf.reduce_sum(
                input_tensor=tf.abs(attention_maps),
                axis=[1, 2, 3]
            )) for attention_maps in attention_maps_sequence
        ] for attention_maps_sequence in attention_maps_sequence_sequence]

        for i, attention_map_loss_sequence in enumerate(attention_map_loss_sequence_sequence):
            for j, attention_map_loss in enumerate(attention_map_loss_sequence):
                tf.summary.scalar("attention_map_loss_sequence_sequence_{}_{}".format(i, j), attention_map_loss)

        attention_map_loss_sequence = [
            tf.reduce_mean(attention_map_loss_sequence)
            for attention_map_loss_sequence in attention_map_loss_sequence_sequence
        ]

        for i, attention_map_loss in enumerate(attention_map_loss_sequence):
            tf.summary.scalar("attention_map_loss_sequence_{}".format(i), attention_map_loss)

        attention_map_loss = tf.reduce_mean(attention_map_loss_sequence)

        tf.summary.scalar("attention_map_loss", attention_map_loss)

        total_variation_loss_sequence_sequence = [[
            tf.reduce_mean(
                input_tensor=tf.image.total_variation(attention_maps)
            ) for attention_maps in attention_maps_sequence
        ] for attention_maps_sequence in attention_maps_sequence_sequence]

        for i, total_variation_loss_sequence in enumerate(total_variation_loss_sequence_sequence):
            for j, total_variation_loss in enumerate(total_variation_loss_sequence):
                tf.summary.scalar("total_variation_loss_sequence_sequence_{}_{}".format(i, j), total_variation_loss)

        total_variation_loss_sequence = [
            tf.reduce_mean(total_variation_loss_sequence)
            for total_variation_loss_sequence in total_variation_loss_sequence_sequence
        ]

        for i, total_variation_loss in enumerate(total_variation_loss_sequence):
            tf.summary.scalar("total_variation_loss_sequence_{}".format(i), total_variation_loss)

        total_variation_loss = tf.reduce_mean(total_variation_loss_sequence)

        tf.summary.scalar("total_variation_loss", total_variation_loss)

        loss_sequence_sequence = [[
            cross_entropy_loss * self.hyper_params.cross_entropy_decay +
            attention_map_loss * self.hyper_params.attention_map_decay +
            total_variation_loss * self.hyper_params.total_variation_decay
            for cross_entropy_loss, attention_map_loss, total_variation_loss
            in zip(cross_entropy_loss_sequence, attention_map_loss_sequence, total_variation_loss_sequence)]
            for cross_entropy_loss_sequence, attention_map_loss_sequence, total_variation_loss_sequence
            in zip(cross_entropy_loss_sequence_sequence, attention_map_loss_sequence_sequence, total_variation_loss_sequence_sequence)]

        for i, loss_sequence in enumerate(loss_sequence_sequence):
            for j, loss in enumerate(loss_sequence):
                tf.summary.scalar("loss_sequence_sequence_{}_{}".format(i, j), loss)

        loss_sequence = [
            tf.reduce_mean(loss_sequence)
            for loss_sequence in loss_sequence_sequence
        ]

        for i, loss in enumerate(loss_sequence):
            tf.summary.scalar("loss_sequence_{}".format(i), loss)

        loss = tf.reduce_mean(loss_sequence)

        tf.summary.scalar("loss", loss)

        accuracy_sequence_sequence = [[
            tf.reduce_mean(tf.cast(labels == classes, tf.float32))
            for labels, classes in zip(labels_sequence, classes_sequence)
        ] for labels_sequence, classes_sequence in zip(labels_sequence_sequence, classes_sequence_sequence)]

        for i, accuracy_sequence in enumerate(accuracy_sequence_sequence):
            for j, accuracy in enumerate(accuracy_sequence):
                tf.summary.scalar("accuracy_sequence_sequence_{}_{}".format(i, j), accuracy)

        accuracy_sequence = [
            tf.reduce_mean(accuracy_sequence)
            for accuracy_sequence in accuracy_sequence_sequence
        ]

        for i, accuracy in enumerate(accuracy_sequence):
            tf.summary.scalar("accuracy_sequence_{}".format(i), accuracy)

        accuracy = tf.reduce_mean(accuracy_sequence)

        tf.summary.scalar("accuracy", accuracy)

        tf.identity(accuracy[0], "non_streaming_accuracy")

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
