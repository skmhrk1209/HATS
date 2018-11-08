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

        images.set_shape([self.hyper_params.batch_size] + images.shape.as_list()[1:])
        labels.set_shape([self.hyper_params.batch_size] + labels.shape.as_list()[1:])

        feature_maps = self.convolutional_network(
            inputs=images,
            training=mode == tf.estimator.ModeKeys.TRAIN
        )

        attention_maps_sequence = self.attention_network(
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

            input_shape = inputs.shape.as_list()
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

        lstm_cell = tf.nn.rnn_cell.LSTMCell(
            num_units=self.num_classes,
            use_peepholes=True,
            initializer=tf.variance_scaling_initializer(
                scale=2.0,
                mode="fan_in",
                distribution="normal",
            )
        )

        logits_sequence = [
            tf.nn.static_rnn(
                cell=lstm_cell,
                inputs=[feature_vectors] * self.string_length,
                initial_state=lstm_cell.zero_state(
                    batch_size=feature_vectors.shape[0],
                    dtype=tf.float32
                )
            )[0] for feature_vectors in feature_vectors_sequence
        ]

        def to_sparse(dense, blank):

            indices = tf.where(tf.not_equal(dense, tf.constant(blank)))
            values = tf.gather_nd(dense, indices)

            return tf.SparseTensor(
                indices=indices,
                values=values,
                dense_shape=dense.shape
            )

        labels_sequence = [
            to_sparse(dense_labels, self.num_classes - 1)
            for dense_labels in tf.unstack(labels, axis=1)
        ]

        for labels in tf.unstack(labels, axis=1):
            tf.Print(labels, labels[0])

        sequence_length_sequence = [[
            tf.shape(tf.where(tf.not_equal(dense_label, tf.constant(self.num_classes - 1))))[0]
            for dense_label in tf.unstack(dense_labels, axis=0)
        ] for dense_labels in tf.unstack(labels, axis=1)]

        ctc_loss = tf.reduce_mean([
            tf.nn.ctc_loss(
                labels=labels,
                inputs=logits,
                sequence_length=sequence_length,
                preprocess_collapse_repeated=False,
                ctc_merge_repeated=True,
                ignore_longer_outputs_than_inputs=False,
                time_major=True
            ) for labels, logits, sequence_length in zip(labels_sequence, logits_sequence, sequence_length_sequence)
        ])

        error_rate = tf.reduce_mean([
            tf.edit_distance(
                hypothesis=tf.cast(tf.nn.ctc_greedy_decoder(
                    inputs=logits,
                    sequence_length=sequence_length
                )[0][0], tf.int32),
                truth=labels,
                normalize=True
            ) for labels, logits, sequence_length in zip(labels_sequence, logits_sequence, sequence_length_sequence)
        ])

        error_rate = tf.identity(error_rate, name="errot_rate")

        attention_map_loss = tf.reduce_mean([
            tf.reduce_mean(tf.reduce_sum(tf.abs(attention_maps), axis=[1, 2, 3]))
            for attention_maps in attention_maps_sequence
        ])

        total_variation_loss = tf.reduce_mean([
            tf.reduce_mean(tf.image.total_variation(attention_maps))
            for attention_maps in attention_maps_sequence
        ])

        loss = \
            ctc_loss * self.hyper_params.ctc_loss_decay + \
            attention_map_loss * self.hyper_params.attention_map_decay + \
            total_variation_loss * self.hyper_params.total_variation_decay

        # ==========================================================================================
        tf.summary.image("images", images, max_outputs=2)
        [tf.summary.image("merged_attention_maps_sequence_{}".format(i), merged_attention_maps, max_outputs=2)
         for i, merged_attention_maps in enumerate(merged_attention_maps_sequence)]
        tf.summary.scalar("ctc_loss", ctc_loss)
        tf.summary.scalar("attention_map_loss", attention_map_loss)
        tf.summary.scalar("total_variation_loss", total_variation_loss)
        tf.summary.scalar("loss", loss)
        tf.summary.scalar("error_rate", error_rate)
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
                eval_metric_ops={"error_rate": error_rate}
            )
