import tensorflow as tf
import numpy as np
import metrics
from networks import ops
from algorithms import *


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
            training=mode == tf.estimator.ModeKeys.TRAIN
        )

        def spatial_flatten(inputs, data_format):

            inputs_shape = inputs.get_shape().as_list()
            outputs_shape = ([-1, inputs_shape[1], np.prod(inputs_shape[2:])] if data_format == "channels_first" else
                             [-1, np.prod(inputs_shape[1:-1]), inputs_shape[-1]])

            return tf.reshape(inputs, outputs_shape)

        feature_vectors = map_innermost_element(
            function=lambda attention_maps: tf.layers.flatten(tf.matmul(
                a=spatial_flatten(feature_maps, self.data_format),
                b=spatial_flatten(attention_maps, self.data_format),
                transpose_a=False if self.data_format == "channels_first" else True,
                transpose_b=True if self.data_format == "channels_first" else False
            )),
            sequence=attention_maps
        )

        for i, units in enumerate(self.units):

            with tf.variable_scope("dense_block_{}".format(i)):

                feature_vectors = map_innermost_element(
                    function=compose(
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
                    sequence=feature_vectors
                )

        logits = map_innermost_element(
            function=lambda feature_vectors: tf.layers.dense(
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
            sequence=feature_vectors
        )

        predictions = map_innermost_element(
            function=lambda logits: tf.argmax(logits, axis=-1),
            sequence=logits
        )

        attention_maps = map_innermost_element(
            function=lambda attention_maps: tf.reduce_sum(
                input_tensor=attention_maps,
                axis=1 if self.data_format == "channels_first" else 3,
                keepdims=True
            ),
            sequence=attention_maps
        )

        if mode == tf.estimator.ModeKeys.PREDICT:

            while isinstance(predictions, list):

                predictions = map_innermost_list(
                    function=lambda predictions: tf.stack(predictions, axis=1),
                    sequence=predictions
                )

            while isinstance(attention_maps, list):

                attention_maps = map_innermost_list(
                    function=lambda attention_maps: tf.stack(attention_maps, axis=1),
                    sequence=attention_maps
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
                function=lambda labels: tf.unstack(labels, axis=1),
                sequence=labels
            )

        losses = map_innermost_element(
            function=lambda labels_logits: tf.losses.sparse_softmax_cross_entropy(
                labels=labels_logits[0],
                logits=labels_logits[1]
            ),
            sequence=zip_innermost_element(labels, logits)
        )

        loss = tf.reduce_mean(losses)

        if self.data_format == "channels_first":

            images = tf.transpose(images, [0, 2, 3, 1])

            attention_maps = map_innermost_element(
                function=lambda attention_maps: tf.transpose(attention_maps, [0, 2, 3, 1]),
                sequence=attention_maps
            )

        tf.summary.image("images", images, max_outputs=2)

        map_innermost_element(
            function=lambda indices_attention_maps: tf.summary.image(
                name="attention_maps_{}".format("_".join(map(str, indices_attention_maps[0]))),
                tensor=indices_attention_maps[1],
                max_outputs=2
            ),
            sequence=enumerate_innermost_element(attention_maps)
        )

        map_innermost_element(
            function=lambda indices_loss: tf.summary.scalar(
                name="loss_{}".format("_".join(map(str, indices_loss[0]))),
                tensor=indices_loss[1]
            ),
            sequence=enumerate_innermost_element(losses)
        )

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

            labels = tf.concat(flatten_innermost_element(map_innermost_list(
                function=lambda labels: tf.stack(labels, axis=1),
                sequence=labels
            )), axis=0)

            logits = tf.concat(flatten_innermost_element(map_innermost_list(
                function=lambda logits: tf.stack(logits, axis=1),
                sequence=logits
            )), axis=0)

            indices = tf.not_equal(labels, self.classes - 1)
            indices = tf.where(tf.reduce_any(indices, axis=1))

            labels = tf.gather_nd(labels, indices)
            logits = tf.gather_nd(logits, indices)

            word_accuracy = metrics.word_accuracy(
                labels=labels,
                logits=logits
            )

            edit_distance = metrics.edit_distance(
                labels=labels,
                logits=logits,
                normalize=True
            )

            return tf.estimator.EstimatorSpec(
                mode=mode,
                loss=loss,
                eval_metric_ops=dict(
                    word_accuracy=word_accuracy,
                    edit_distance=edit_distance
                )
            )
