import tensorflow as tf
import numpy as np
import metrics
from algorithms import *


def spatial_shape(inputs, channels_first):

    inputs_shape = inputs.get_shape().as_list()

    return inputs_shape[2:] if channels_first else inputs_shape[1:-1]


def spatial_flatten(inputs, channels_first):

    inputs_shape = inputs.get_shape().as_list()
    outputs_shape = ([-1, inputs_shape[1], np.prod(inputs_shape[2:])] if channels_first else
                     [-1, np.prod(inputs_shape[1:-1]), inputs_shape[-1]])

    return tf.reshape(inputs, outputs_shape)


class Model(object):

    class AccuracyType:
        FULL_SEQUENCE, EDIT_DISTANCE = range(2)

    def __init__(self, convolutional_network, attention_network,
                 num_classes, channels_first, accuracy_type, hyper_params):

        self.convolutional_network = convolutional_network
        self.attention_network = attention_network
        self.num_classes = num_classes
        self.channels_first = channels_first
        self.data_format = "channels_first" if channels_first else "channels_last"
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

        feature_vectors = map_innermost_element(
            function=lambda attention_maps: tf.layers.flatten(tf.matmul(
                a=spatial_flatten(feature_maps, self.channels_first),
                b=spatial_flatten(attention_maps, self.channels_first),
                transpose_a=False if self.channels_first else True,
                transpose_b=True if self.channels_first else False
            )),
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

        predictions = map_innermost_element(
            function=lambda logits: tf.argmax(
                input=logits,
                axis=-1
            ),
            sequence=logits
        )

        attention_maps = map_innermost_element(
            function=lambda attention_maps: tf.reduce_sum(
                input_tensor=attention_maps,
                axis=1 if self.channels_first else 3,
                keep_dims=True
            ),
            sequence=attention_maps
        )

        if mode == tf.estimator.ModeKeys.PREDICT:

            while isinstance(attention_maps, list):

                attention_maps = map_innermost_list(
                    function=lambda attention_maps: tf.stack(attention_maps, axis=1),
                    sequence=attention_maps
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
                    attention_maps=attention_maps,
                    predictions=predictions
                )
            )

        while all(flatten_innermost_element(map_innermost_element(lambda labels: len(labels.shape) > 1, labels))):

            labels = map_innermost_element(
                function=lambda labels: tf.unstack(labels, axis=1),
                sequence=labels
            )

        loss = tf.reduce_mean(map_innermost_element(
            function=lambda logits_labels: tf.losses.sparse_softmax_cross_entropy(
                logits=logits_labels[0],
                labels=logits_labels[1]
            ),
            sequence=zip_innermost_element(logits, labels)
        ))

        loss += tf.reduce_mean(map_innermost_element(
            function=lambda attention_maps: tf.reduce_mean(tf.reduce_sum(attention_maps, axis=[1, 2, 3])),
            sequence=attention_maps
        )) * self.hyper_params.attention_map_decay

        # ==========================================================================================
        tf.summary.image("images", images, max_outputs=2)

        map_innermost_element(
            function=lambda indices_attention_maps: tf.summary.image(
                name="indices_attention_maps_{}".format("_".join(map(str, indices_attention_maps[0]))),
                tensor=indices_attention_maps[1],
                max_outputs=2
            ),
            sequence=enumerate_innermost_element(attention_maps)
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
