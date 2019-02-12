import tensorflow as tf
import numpy as np
import functools
import metrics
import summary
from networks import ops
from algorithms import *


def seq_len_getter(labels, classes, indices):

    begin = [0] + indices + [0] * (len(labels.shape[1:]) - len(indices))
    size = [-1] + [1] * len(indices) + [-1] * (len(labels.shape[1:]) - len(indices))

    labels = tf.slice(labels, begin, size)

    return tf.count_nonzero(
        input_tensor=tf.reduce_any(
            input_tensor=tf.not_equal(labels, classes),
            axis=list(range(2, len(labels.shape)))
        ),
        axis=1
    )


def spatial_flatten(inputs, data_format):

    inputs_shape = inputs.shape.as_list()
    outputs_shape = ([-1, inputs_shape[1], np.prod(inputs_shape[2:])] if data_format == "channels_first" else
                     [-1, np.prod(inputs_shape[1:-1]), inputs_shape[-1]])

    return tf.reshape(inputs, outputs_shape)


class HATS(object):

    def __init__(self, backbone_network, attention_network,
                 units, classes, data_format, hyper_params):

        self.backbone_network = backbone_network
        self.attention_network = attention_network
        self.units = units
        self.classes = classes
        self.data_format = data_format
        self.hyper_params = hyper_params

    def __call__(self, images, labels, mode):

        feature_maps = self.backbone_network(
            inputs=images,
            training=mode == tf.estimator.ModeKeys.TRAIN
        )

        attention_maps = self.attention_network(
            inputs=feature_maps,
            seq_len_getter=functools.partial(
                seq_len_getter,
                labels=labels,
                classes=self.classes
            ),
            training=mode == tf.estimator.ModeKeys.TRAIN
        )

        feature_vectors = map_innermost_element(
            func=lambda attention_maps: tf.layers.flatten(tf.matmul(
                a=spatial_flatten(feature_maps, self.data_format),
                b=spatial_flatten(attention_maps, self.data_format),
                transpose_a=False if self.data_format == "channels_first" else True,
                transpose_b=True if self.data_format == "channels_first" else False
            )),
            seq=attention_maps
        )

        for i, units in enumerate(self.units):

            with tf.variable_scope("dense_block_{}".format(i)):

                feature_vectors = map_innermost_element(
                    func=compose(
                        lambda inputs: tf.layers.dense(
                            inputs=inputs,
                            units=units,
                            use_bias=False,
                            kernel_initializer=tf.initializers.variance_scaling(
                                scale=2.0,
                                mode="fan_in",
                                distribution="untruncated_normal"
                            ),
                            name="dense",
                            reuse=tf.AUTO_REUSE
                        ),
                        lambda inputs: ops.batch_normalization(
                            inputs=inputs,
                            data_format=self.data_format,
                            training=mode == tf.estimator.ModeKeys.TRAIN,
                            name="batch_normalization",
                            reuse=tf.AUTO_REUSE
                        ),
                        lambda inputs: tf.nn.relu(inputs)
                    ),
                    seq=feature_vectors
                )

        logits = map_innermost_element(
            func=lambda feature_vectors: tf.layers.dense(
                inputs=feature_vectors,
                units=self.classes,
                kernel_initializer=tf.initializers.variance_scaling(
                    scale=1.0,
                    mode="fan_avg",
                    distribution="untruncated_normal"
                ),
                bias_initializer=tf.initializers.zeros(),
                name="logits",
                reuse=tf.AUTO_REUSE
            ),
            seq=feature_vectors
        )

        predictions = map_innermost_element(
            func=lambda logits: tf.argmax(logits, axis=-1),
            seq=logits
        )

        attention_maps = map_innermost_element(
            func=lambda indices_attention_maps: tf.reduce_sum(
                input_tensor=indices_attention_maps[1],
                axis=1 if self.data_format == "channels_first" else 3,
                keepdims=True,
                name="attention_maps_{}".format("_".join(map(str, indices_attention_maps[0])))
            ),
            seq=enumerate_innermost_element(attention_maps)
        )

        if mode == tf.estimator.ModeKeys.PREDICT:

            while isinstance(predictions, list):

                predictions = map_innermost_list(
                    func=lambda predictions: tf.stack(predictions, axis=1),
                    seq=predictions
                )

            while isinstance(attention_maps, list):

                attention_maps = map_innermost_list(
                    func=lambda attention_maps: tf.stack(attention_maps, axis=1),
                    seq=attention_maps
                )

            return tf.estimator.EstimatorSpec(
                mode=mode,
                predictions=dict(
                    images=images,
                    predictions=predictions,
                    attention_maps=attention_maps
                )
            )

        while all(flatten_innermost_element(map_innermost_element(lambda labels: len(labels.shape) > 1, labels))):

            labels = map_innermost_element(
                func=lambda labels: tf.unstack(labels, axis=1),
                seq=labels
            )

        labels = tf.concat(flatten_innermost_element(map_innermost_list(
            func=lambda labels: tf.stack(labels, axis=1),
            seq=labels
        )), axis=0)

        logits = tf.concat(flatten_innermost_element(map_innermost_list(
            func=lambda logits: tf.stack(logits, axis=1),
            seq=logits
        )), axis=0)

        predictions = tf.concat(flatten_innermost_element(map_innermost_list(
            func=lambda predictions: tf.stack(predictions, axis=1),
            seq=predictions
        )), axis=0)

        indices = tf.reduce_any(tf.less(labels, self.classes - 1), axis=1)
        indices = tf.where(indices)

        labels = tf.gather_nd(labels, indices)
        logits = tf.gather_nd(logits, indices)

        sequence_lengths = tf.count_nonzero(tf.less(labels, self.classes), axis=1)
        sequence_mask = tf.sequence_mask(sequence_lengths, labels.shape[1], dtype=tf.int32)

        loss = tf.contrib.seq2seq.sequence_loss(
            logits=logits,
            targets=labels,
            weights=tf.cast(sequence_mask, tf.float32),
            average_across_timesteps=True,
            average_across_batch=True
        )

        labels *= sequence_mask
        predictions *= sequence_mask

        word_accuracy = metrics.word_accuracy(
            labels=labels,
            predictions=predictions,
            name="word_accuracy"
        )

        summary.any(word_accuracy)
        summary.any(images, data_format=self.data_format, max_outputs=2)
        for attention_maps in flatten_innermost_element(attention_maps):
            summary.any(attention_maps, data_format=self.data_format, max_outputs=2)

        if mode == tf.estimator.ModeKeys.TRAIN:

            with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):

                train_op = self.hyper_params.optimizer.minimize(
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
                eval_metric_ops=dict(
                    word_accuracy=word_accuracy
                )
            )
