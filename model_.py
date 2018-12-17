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

    def __init__(self, convolutional_network, rnn_params,
                 num_classes, channels_first, accuracy_type, hyper_params):

        self.convolutional_network = convolutional_network
        self.rnn_params = rnn_params
        self.num_classes = num_classes
        self.channels_first = channels_first
        self.data_format = "channels_first" if channels_first else "channels_last"
        self.accuracy_type = accuracy_type
        self.hyper_params = hyper_params

    def __call__(self, features, labels, mode):

        inputs = features["image"]

        inputs = self.convolutional_network(
            inputs=inputs,
            training=mode == tf.estimator.ModeKeys.TRAIN
        )

        inputs = tf.layers.flatten(inputs)

        for i, rnn_param in enumerate(self.rnn_params):

            with tf.variable_scope("rnn_block_{}".format(i)):

                multi_cell = tf.nn.rnn_cell.MultiRNNCell([
                    tf.nn.rnn_cell.LSTMCell(
                        num_units=num_units,
                        use_peepholes=True
                    ) for num_units in rnn_param.num_units
                ])

                inputs = map_innermost_element(
                    function=lambda inputs: tf.nn.static_rnn(
                        cell=multi_cell,
                        inputs=[inputs] * rnn_param.sequence_length,
                        initial_state=multi_cell.zero_state(
                            batch_size=tf.shape(inputs)[0],
                            dtype=tf.float32
                        ),
                        scope="rnn"
                    )[0],
                    sequence=inputs
                )

        logits = map_innermost_element(
            function=lambda inputs: tf.layers.dense(
                inputs=inputs,
                units=self.num_classes,
                name="logits",
                reuse=tf.AUTO_REUSE
            ),
            sequence=inputs
        )

        predictions = map_innermost_element(
            function=lambda logits: tf.argmax(
                input=logits,
                axis=-1
            ),
            sequence=logits
        )

        if mode == tf.estimator.ModeKeys.PREDICT:

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
