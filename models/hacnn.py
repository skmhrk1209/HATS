import tensorflow as tf
import numpy as np


class Model(object):

    def __init__(self, convolutional_network, attention_network,
                 string_length, lstm_units, num_classes, data_format, hyper_params):

        self.convolutional_network = convolutional_network
        self.attention_network = attention_network
        self.string_length = string_length
        self.lstm_units = lstm_units
        self.num_classes = num_classes
        self.data_format = data_format
        self.hyper_params = hyper_params

    def __call__(self, features, labels, mode):

        images = features["image"]

        feature_maps = self.convolutional_network(
            inputs=images,
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

        def flatten_images(inputs, data_format):

            input_shape = inputs.get_shape().as_list()
            output_shape = ([-1, input_shape[1], np.prod(input_shape[2:4])] if self.data_format == "channels_first" else
                            [-1, np.prod(input_shape[1:3]), input_shape[3]])

            return tf.reshape(inputs, output_shape)

        feature_vectors_sequence_sequence = [[
            tf.layers.flatten(tf.matmul(
                a=flatten_images(feature_maps, self.data_format),
                b=flatten_images(attention_maps, self.data_format),
                transpose_a=False if self.data_format == "channels_first" else True,
                transpose_b=True if self.data_format == "channels_first" else False
            )) for attention_maps in attention_maps_sequence
        ] for attention_maps_sequence in attention_maps_sequence_sequence]

        logits_sequence_sequence = [[
            tf.layers.dense(
                inputs=feature_vectors,
                units=self.num_classes,
                name="logits",
                reuse=tf.AUTO_REUSE
            ) for feature_vectors in feature_vectors_sequence
        ] for feature_vectors_sequence in feature_vectors_sequence_sequence]

        predictions_sequence_sequence = [[
            tf.argmax(logits, axis=-1, output_type=tf.int32)
            for logits in logits_sequence
        ] for logits_sequence in logits_sequence_sequence]

        labels_sequence_sequence = [
            tf.unstack(labels, axis=1)
            for labels in tf.unstack(labels, axis=1)
        ]

        if mode == tf.estimator.ModeKeys.PREDICT:

            return tf.estimator.EstimatorSpec(
                mode=mode,
                predictions={}
            )

        cross_entropy_loss = tf.reduce_mean([[
            tf.losses.sparse_softmax_cross_entropy(
                labels=labels,
                logits=logits
            ) for labels, logits in zip(labels_sequence, logits_sequence)
        ] for labels_sequence, logits_sequence in zip(labels_sequence_sequence, logits_sequence_sequence)])

        attention_map_loss = tf.reduce_mean([[
            tf.reduce_mean(tf.reduce_sum(tf.abs(attention_maps), axis=[1, 2, 3]))
            for attention_maps in attention_maps_sequence
        ] for attention_maps_sequence in attention_maps_sequence_sequence])

        total_variation_loss = tf.reduce_mean([[
            tf.reduce_mean(tf.image.total_variation(attention_maps))
            for attention_maps in attention_maps_sequence
        ] for attention_maps_sequence in attention_maps_sequence_sequence])

        loss = \
            cross_entropy_loss * self.hyper_params.cross_entropy_decay + \
            attention_map_loss * self.hyper_params.attention_map_decay + \
            total_variation_loss * self.hyper_params.total_variation_decay

        labels = tf.stack([
            tf.stack(labels_sequence, axis=1)
            for labels_sequence in labels_sequence_sequence
        ], axis=1)

        predictions = tf.stack([
            tf.stack(predictions_sequence, axis=1)
            for predictions_sequence in predictions_sequence_sequence
        ], axis=1)

        streaming_accuracy = tf.metrics.accuracy(
            labels=labels,
            predictions=predictions
        )

        non_streaming_accuracy = tf.reduce_mean(
            input_tensor=tf.cast(tf.equal(labels, predictions), tf.float32),
            name="non_streaming_accuracy"
        )

        # ==========================================================================================
        tf.summary.image("images", images, max_outputs=2)
        [tf.summary.image("merged_attention_maps_sequence_{}".format(i), merged_attention_maps, max_outputs=2)
         for i, merged_attention_maps in enumerate(merged_attention_maps_sequence)]
        tf.summary.scalar("cross_entropy_loss", cross_entropy_loss)
        tf.summary.scalar("attention_map_loss", attention_map_loss)
        tf.summary.scalar("total_variation_loss", total_variation_loss)
        tf.summary.scalar("loss", loss)
        tf.summary.scalar("accuracy", streaming_accuracy[1])
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
                eval_metric_ops={"streaming_accuracy": streaming_accuracy}
            )
