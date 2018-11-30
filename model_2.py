import tensorflow as tf
import numpy as np
import metrics
from algorithms import *


class Model(object):

    class AccuracyType:
        FULL_SEQUENCE, EDIT_DISTANCE = range(2)

    def __init__(self, convolutional_network, attention_network,
                 num_classes, data_format, accuracy_type, hyper_params):

        self.convolutional_network = convolutional_network
        self.attention_network = attention_network
        self.num_classes = num_classes
        self.data_format = data_format
        self.accuracy_type = accuracy_type
        self.hyper_params = hyper_params

    def __call__(self, features, labels, mode):

        images = features["image"]

        feature_maps = self.convolutional_network(
            inputs=images,
            training=mode == tf.estimator.ModeKeys.TRAIN
        )

        attention_maps = self.attention_network(
            inputs=feature_maps,
            training=mode == tf.estimator.ModeKeys.TRAIN
        )

        merged_attention_maps = map_innermost_element(
            function=lambda attention_maps: tf.reduce_sum(
                input_tensor=attention_maps,
                axis=1 if self.data_format == "channels_first" else 3,
                keep_dims=True
            ),
            sequence=attention_maps
        )

        def global_average_pooling2d(inputs, data_format):

            return tf.reduce_mean(
                input_tensor=inputs,
                axis=[2, 3] if data_format == "channels_first" else [1, 2]
            )

        map_innermost_element(
            function=lambda attention_maps: print(attention_maps.shape),
            sequence=attention_maps
        )

        feature_vectors = map_innermost_element(
            function=lambda attention_maps: global_average_pooling2d(
                inputs=tf.multiply(feature_maps, attention_maps),
                data_format=self.data_format
            ),
            sequence=attention_maps
        )

        logits = map_innermost_element(
            function=lambda feature_vectors: tf.layers.dense(
                inputs=feature_vectors,
                units=self.num_classes,
                name="logits",
                reuse=tf.AUTO_REUSE
            ),
            sequence=feature_vectors
        )

        if mode == tf.estimator.ModeKeys.PREDICT:

            predictions = map_innermost_element(
                function=lambda logits: tf.argmax(
                    input=logits,
                    axis=1
                ),
                sequence=logits
            )

            while isinstance(merged_attention_maps, list):

                merged_attention_maps = map_innermost_list(
                    function=lambda merged_attention_maps: tf.stack(merged_attention_maps, axis=1),
                    sequence=merged_attention_maps
                )

            while isinstance(predictions, list):

                predictions = map_innermost_list(
                    function=lambda predictions: tf.stack(predictions, axis=1),
                    sequence=predictions
                )

            return tf.estimator.EstimatorSpec(
                mode=mode,
                predictions=dict(
                    images=images,
                    merged_attention_maps=merged_attention_maps,
                    predictions=predictions
                )
            )

        while all(flatten_innermost_element(map_innermost_element(lambda labels: len(labels.shape) > 1, labels))):

            labels = map_innermost_element(
                function=lambda labels: tf.unstack(labels, axis=1),
                sequence=labels
            )

        cross_entropy_losses = map_innermost_element(
            function=lambda logits_labels: tf.losses.sparse_softmax_cross_entropy(
                logits=logits_labels[0],
                labels=logits_labels[1]
            ),
            sequence=zip_innermost_element(logits, labels)
        )

        attention_map_losses = map_innermost_element(
            function=lambda attention_maps: tf.reduce_mean(
                tf.reduce_sum(tf.abs(attention_maps), axis=[1, 2, 3])
            ),
            sequence=attention_maps
        )

        losses = map_innermost_element(
            function=lambda cross_entropy_loss_attention_map_loss: (
                cross_entropy_loss_attention_map_loss[0] * self.hyper_params.cross_entropy_decay +
                cross_entropy_loss_attention_map_loss[1] * self.hyper_params.attention_map_decay
            ),
            sequence=zip_innermost_element(cross_entropy_losses, attention_map_losses)
        )

        loss = tf.reduce_mean(losses)

        # ==========================================================================================
        tf.summary.image("images", images, max_outputs=2)

        map_innermost_element(
            function=lambda indices_merged_attention_maps: tf.summary.image(
                name="merged_attention_maps_{}".format("_".join(map(str, indices_merged_attention_maps[0]))),
                tensor=indices_merged_attention_maps[1],
                max_outputs=2
            ),
            sequence=enumerate_innermost_element(merged_attention_maps)
        )

        map_innermost_element(
            function=lambda indices_cross_entropy_loss: tf.summary.scalar(
                name="cross_entropy_loss_{}".format("_".join(map(str, indices_cross_entropy_loss[0]))),
                tensor=indices_cross_entropy_loss[1]
            ),
            sequence=enumerate_innermost_element(cross_entropy_losses)
        )

        map_innermost_element(
            function=lambda indices_attention_map_loss: tf.summary.scalar(
                name="attention_map_loss_{}".format("_".join(map(str, indices_attention_map_loss[0]))),
                tensor=indices_attention_map_loss[1]
            ),
            sequence=enumerate_innermost_element(attention_map_losses)
        )

        map_innermost_element(
            function=lambda indices_loss: tf.summary.scalar(
                name="loss_{}".format("_".join(map(str, indices_loss[0]))),
                tensor=indices_loss[1]
            ),
            sequence=enumerate_innermost_element(losses)
        )
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

            accuracy_functions = {
                Model.AccuracyType.FULL_SEQUENCE: metrics.full_sequence_accuracy,
                Model.AccuracyType.EDIT_DISTANCE: metrics.edit_distance_accuracy,
            }

            accuracies = map_innermost_element(
                function=lambda logits_labels: accuracy_functions[self.accuracy_type](
                    logits=tf.stack(logits_labels[0], axis=1),
                    labels=tf.stack(logits_labels[1], axis=1),
                    time_major=False
                ),
                sequence=zip_innermost_list(logits, labels)
            )

            return tf.estimator.EstimatorSpec(
                mode=mode,
                loss=loss,
                eval_metric_ops=dict(flatten_innermost_element(map_innermost_element(
                    function=lambda indices_accuracy: (
                        "accuracy_{}".format("_".join(map(str, indices_accuracy[0]))),
                        indices_accuracy[1]
                    ),
                    sequence=enumerate_innermost_element(accuracies)
                )))
            )
