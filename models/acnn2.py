import tensorflow as tf
import numpy as np


class Model(object):

    def __init__(self, global_attention_network, local_attention_network,
                 convolutional_network, data_format, hyper_params):

        self.global_attention_network = global_attention_network
        self.local_attention_network = local_attention_network
        self.convolutional_network = convolutional_network
        self.data_format = data_format
        self.hyper_params = hyper_params

    def __call__(self, features, labels, mode):

        images = features["images"]

        tf.summary.image("images", images, max_outputs=2)

        global_attention_maps_seq = self.global_attention_network(
            inputs=images,
            training=mode == tf.estimator.ModeKeys.TRAIN,
            name="global_attention_network",
            reuse=tf.AUTO_REUSE
        )

        for i, global_attention_maps in enumerate(global_attention_maps_seq):
            tf.summary.image("global_attention_maps_{}".format(i), global_attention_maps, max_outputs=2)

        images_seq = [
            images * global_attention_maps
            for global_attention_maps in global_attention_maps_seq
        ]

        local_attention_maps_seq_seq = [
            self.local_attention_network(
                inputs=images,
                training=mode == tf.estimator.ModeKeys.TRAIN,
                name="local_attention_network",
                reuse=tf.AUTO_REUSE
            ) for images in images_seq
        ]

        for i, local_attention_maps_seq in enumerate(local_attention_maps_seq_seq):
            for j, local_attention_maps in enumerate(local_attention_maps_seq):
                tf.summary.image("local_attention_maps_{}_{}".format(i, j), local_attention_maps, max_outputs=2)

        images_seq_seq = [[
            images * attention_maps
            for attention_maps in local_attention_maps_seq
        ] for images, local_attention_maps_seq in zip(images_seq, local_attention_maps_seq_seq)]

        logits_seq_seq = [[
            self.convolutional_network(
                inputs=images,
                training=mode == tf.estimator.ModeKeys.TRAIN,
                name="convolutional_network",
                reuse=tf.AUTO_REUSE
            ) for images in images_seq
        ] for images_seq in images_seq_seq]

        classes_seq_seq = [[
            tf.argmax(
                input=logits,
                axis=-1
            ) for logits in logits_seq
        ] for logits_seq in logits_seq_seq]

        print(len(logits_seq_seq), len(logits_seq_seq[0]))

        if mode == tf.estimator.ModeKeys.PREDICT:

            features.update({
                "global_attention_maps_seq": tf.stack(
                    global_attention_maps_seq,
                    axis=1
                ),
                "local_attention_maps_seq_seq": tf.stack([
                    tf.stack(local_attention_maps_seq, axis=1)
                    for local_attention_maps_seq in local_attention_maps_seq
                ], axis=1),
                "classes_seq_seq": tf.stack([
                    tf.stack(classes_seq, axis=1)
                    for classes_seq in classes_seq_seq
                ], axis=1)
            })

            return tf.estimator.EstimatorSpec(
                mode=mode,
                predictions=features
            )

        labels_seq_seq = [
            tf.unstack(multi_labels, axis=1)
            for multi_labels in tf.unstack(labels, axis=1)
        ]

        cross_entropy_loss = tf.reduce_mean([[
            tf.losses.sparse_softmax_cross_entropy(
                labels=labels,
                logits=logits
            ) for labels, logits in zip(labels_seq, logits_seq)
        ] for labels_seq, logits_seq in zip(labels_seq_seq, logits_seq_seq)])

        tf.summary.scalar("cross_entropy_loss", cross_entropy_loss)

        global_attention_map_loss = tf.reduce_mean([
            tf.reduce_mean(tf.reduce_sum(
                input_tensor=tf.abs(global_attention_maps),
                axis=[1, 2, 3]
            )) for global_attention_maps in global_attention_maps_seq
        ])

        tf.summary.scalar("global_attention_map_loss", global_attention_map_loss)

        local_attention_map_loss = tf.reduce_mean([[
            tf.reduce_mean(tf.reduce_sum(
                input_tensor=tf.abs(local_attention_maps),
                axis=[1, 2, 3]
            )) for local_attention_maps in local_attention_maps_seq
        ] for local_attention_maps_seq in local_attention_maps_seq_seq])

        tf.summary.scalar("local_attention_map_loss", local_attention_map_loss)

        global_total_variation_loss = tf.reduce_mean([
            tf.reduce_mean(
                input_tensor=tf.image.total_variation(global_attention_maps)
            ) for global_attention_maps in global_attention_maps_seq
        ])

        tf.summary.scalar("global_total_variation_loss", global_total_variation_loss)

        local_total_variation_loss = tf.reduce_mean([[
            tf.reduce_mean(
                input_tensor=tf.image.total_variation(local_attention_maps)
            ) for local_attention_maps in local_attention_maps_seq
        ] for local_attention_maps_seq in local_attention_maps_seq_seq])

        tf.summary.scalar("local_total_variation_loss", local_total_variation_loss)

        loss = \
            cross_entropy_loss * self.hyper_params.cross_entropy_decay + \
            global_attention_map_loss * self.hyper_params.global_attention_map_decay + \
            local_attention_map_loss * self.hyper_params.local_attention_map_decay + \
            global_total_variation_loss * self.hyper_params.global_total_variation_decay + \
            local_total_variation_loss * self.hyper_params.local_total_variation_decay

        tf.summary.scalar("loss", loss)

        streaming_accuracy = tf.metrics.accuracy(
            labels=labels_seq_seq,
            predictions=classes_seq_seq
        )

        tf.summary.scalar("streaming_accuracy", streaming_accuracy[1])

        tf.identity(streaming_accuracy[0], "streaming_accuracy_value")

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
