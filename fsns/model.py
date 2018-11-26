import tensorflow as tf
import numpy as np
import metrics
from algorithms import *


class Model(object):

    class AccuracyType:
        FULL_SEQUENCE, EDIT_DISTANCE = range(2)

    def __init__(self, convolutional_network, attention_network,
                 num_classes, num_tiles, data_format, accuracy_type, hyper_params):

        self.convolutional_network = convolutional_network
        self.attention_network = attention_network
        self.num_classes = num_classes
        self.num_tiles = num_tiles
        self.data_format = data_format
        self.accuracy_type = accuracy_type
        self.hyper_params = hyper_params

    def __call__(self, features, labels, mode):

        images = features["image"]

        images = tf.split(
            value=images,
            num_or_size_splits=self.num_tiles,
            axis=3 if self.data_format == "channels_first" else 2
        )

        feature_maps = map_innermost_element(
            function=lambda images: self.convolutional_network(
                inputs=images,
                training=mode == tf.estimator.ModeKeys.TRAIN,
                name="convolutional_network",
                reuse=tf.AUTO_REUSE
            ),
            sequence=images
        )

        attention_maps = map_innermost_element(
            function=lambda feature_maps: self.attention_network(
                inputs=feature_maps,
                training=mode == tf.estimator.ModeKeys.TRAIN,
                reuse=tf.AUTO_REUSE
            ),
            sequence=feature_maps
        )

        merged_attention_maps = map_innermost_element(
            function=lambda attention_maps: tf.reduce_sum(
                input_tensor=attention_maps,
                axis=1 if self.data_format == "channels_first" else 3,
                keep_dims=True
            ),
            sequence=attention_maps
        )

        def flatten_images(inputs, data_format):

            input_shape = inputs.get_shape().as_list()
            output_shape = ([-1, input_shape[1], np.prod(input_shape[2:4])] if self.data_format == "channels_first" else
                            [-1, np.prod(input_shape[1:3]), input_shape[3]])

            return tf.reshape(inputs, output_shape)

        feature_vectors = map_innermost_element(
            function=lambda feature_maps_atention_maps: map_innermost_element(
                function=lambda attention_maps: tf.layers.flatten(tf.matmul(
                    a=flatten_images(feature_maps_atention_maps[0], self.data_format),
                    b=flatten_images(attention_maps, self.data_format),
                    transpose_a=False if self.data_format == "channels_first" else True,
                    transpose_b=True if self.data_format == "channels_first" else False
                )),
                sequence=feature_maps_atention_maps[1]
            ),
            sequence=zip_innermost_element(feature_maps, attention_maps)
        )

        feature_vectors = map_innermost_element(
            function=lambda feature_vectors: tf.concat(
                values=feature_vectors,
                axis=1
            ),
            sequence=zip_innermost_element(*feature_vectors)
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

        predictions = map_innermost_element(
            function=lambda logits: tf.argmax(
                input=logits,
                axis=1,
                output_type=tf.int32
            ),
            sequence=logits
        )

        if mode == tf.estimator.ModeKeys.PREDICT:

            while isinstance(images, list):

                images = map_innermost_list(
                    function=lambda images: tf.stack(images, axis=1),
                    sequence=images
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
            function=lambda attention_maps: tf.reduce_mean([
                tf.reduce_mean(
                    tf.reduce_sum(tf.abs(attention_maps), axis=[1, 2, 3])
                ) for attention_maps in attention_maps
            ]),
            sequence=zip_innermost_element(*attention_maps)
        )

        losses = map_innermost_element(
            function=lambda cross_entropy_loss_attention_map_loss: (
                cross_entropy_loss_attention_map_loss[0] * self.hyper_params.cross_entropy_decay +
                cross_entropy_loss_attention_map_loss[1] * self.hyper_params.attention_map_decay
            ),
            sequence=zip_innermost_element(cross_entropy_losses, attention_map_losses)
        )

        loss = tf.reduce_mean(losses)

        '''
        logits = map_innermost_list(
            function=lambda logits: tf.stack(logits, axis=1),
            sequence=logits
        )

        labels = map_innermost_list(
            function=lambda labels: tf.stack(labels, axis=1),
            sequence=labels
        )
        '''

        accuracy_function = {
            Model.AccuracyType.FULL_SEQUENCE: metrics.full_sequence_accuracy,
            Model.AccuracyType.EDIT_DISTANCE: metrics.edit_distance_accuracy,
        }

        accuracies = map_innermost_element(
            function=lambda logits_labels: accuracy_function[self.accuracy_type](
                logits=tf.stack(logits_labels[0], axis=1),
                labels=tf.stack(logits_labels[1], axis=1),
                time_major=False
            ),
            sequence=zip_innermost_list(logits, labels)
        )

        map_innermost_element(
            function=lambda accuracy: tf.add_to_collection(tf.GraphKeys.UPDATE_OPS, accuracy[1]),
            sequence=accuracies
        )

        accuracy = tf.reduce_mean(map_innermost_element(
            function=lambda accuracy: accuracy[0],
            sequence=accuracies
        )), tf.no_op()

        # ==========================================================================================
        map_innermost_element(
            function=lambda indices_images: tf.summary.image(
                name="images_{}".format("_".join(map(str, indices_images[0]))),
                tensor=indices_images[1],
                max_outputs=2
            ),
            sequence=enumerate_innermost_element(images)
        )

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

        map_innermost_element(
            function=lambda indices_accuracy: tf.summary.scalar(
                name="accuracy_{}".format("_".join(map(str, indices_accuracy[0]))),
                tensor=indices_accuracy[1][0]
            ),
            sequence=enumerate_innermost_element(accuracies)
        )

        tf.summary.scalar("accuracy_", accuracy[0])

        tf.identity(accuracy[0], "accuracy_")
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
                eval_metric_ops={
                    **dict(accuracy=accuracy),
                    **dict(flatten_innermost_element(map_innermost_element(
                        function=lambda indices_accuracy: (
                            "accuracy_{}".format("_".join(map(str, indices_accuracy[0]))),
                            indices_accuracy[1]
                        ),
                        sequence=enumerate_innermost_element(accuracies)
                    )))
                }
            )
