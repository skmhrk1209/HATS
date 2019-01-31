import tensorflow as tf
import numpy as np
import metrics
from algorithms import *


class HATS(object):

    def __init__(self, backbone_network, attention_network, num_classes, data_format, hyper_params):

        self.backbone_network = backbone_network
        self.attention_network = attention_network
        self.num_classes = num_classes
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
            function=lambda logits: tf.argmax(logits, axis=-1),
            sequence=logits
        )

        attention_maps = map_innermost_element(
            function=lambda attention_maps: tf.reduce_sum(
                input_tensor=attention_maps,
                axis=1 if self.data_format == "channels_first" else 3,
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
            function=lambda labels_logits: tf.losses.sparse_softmax_cross_entropy(*labels_logits),
            sequence=zip_innermost_element(labels, logits)
        ))

        loss += tf.reduce_mean(map_innermost_element(
            function=lambda attention_maps: tf.reduce_mean(tf.reduce_sum(attention_maps, axis=[1, 2, 3])),
            sequence=attention_maps
        )) * self.hyper_params.attention_decay

        labels = tf.concat(flatten_innermost_element(map_innermost_list(
            function=lambda labels: tf.stack(labels, axis=1),
            sequence=labels
        )), axis=0)

        logits = tf.concat(flatten_innermost_element(map_innermost_list(
            function=lambda logits: tf.stack(logits, axis=1),
            sequence=logits
        )), axis=0)

        accuracy = metrics.sequence_accuracy(labels, logits)

        print("num params: {}".format(sum([
            np.prod(variable.get_shape().as_list())
            for variable in tf.trainable_variables()
        ])))

        # ==========================================================================================
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
        # ==========================================================================================

        if mode == tf.estimator.ModeKeys.TRAIN:

            with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):

                optimizer = tf.train.AdamOptimizer(
                    learning_rate=self.hyper_params.learning_rate,
                    beta1=self.hyper_params.beta1,
                    beta2=self.hyper_params.beta2
                )

                train_op = optimizer.minimize(
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
                eval_metric_ops=dict(accuracy=accuracy)
            )
