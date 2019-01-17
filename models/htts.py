import tensorflow as tf
import numpy as np
import metrics
from networks import ops
from algorithms import *


class HTTS(object):

    def __init__(self, backbone_network, hierarchical_transformer_network,
                 out_size, num_classes, data_format, hyper_params, pretrained_model_dir=None):

        self.backbone_network = backbone_network
        self.hierarchical_transformer_network = hierarchical_transformer_network
        self.num_classes = num_classes
        self.data_format = data_format
        self.hyper_params = hyper_params
        self.pretrained_model_dir = pretrained_model_dir

    def __call__(self, images, labels, mode):

        feature_maps = self.backbone_network(
            inputs=images,
            training=mode == tf.estimator.ModeKeys.TRAIN
        )

        params = self.hierarchical_transformer_network(
            inputs=feature_maps,
            training=mode == tf.estimator.ModeKeys.TRAIN
        )

        feature_vectors = map_innermost_element(
            function=lambda params: tf.layers.flatten(ops.spatial_transformer(
                inputs=feature_maps,
                params=params,
                out_size=out_size
            )),
            sequence=params
        )

        if self.pretrained_model_dir:

            tf.train.init_from_checkpoint(
                ckpt_dir_or_file=self.pretrained_model_dir,
                assignment_map={"/": "/"}
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
                axis=-1,
                output_type=tf.int32
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

        labels = tf.concat(flatten_innermost_element(map_innermost_list(
            function=lambda labels: tf.stack(labels, axis=1),
            sequence=labels
        )), axis=0)

        logits = tf.concat(flatten_innermost_element(map_innermost_list(
            function=lambda logits: tf.stack(logits, axis=1),
            sequence=logits
        )), axis=0)

        error_rate = metrics.normalized_edit_distance(labels, logits)

        # ==========================================================================================
        if self.data_format == "channels_first":

            images = tf.transpose(images, [0, 2, 3, 1])

        tf.summary.image("images", images, max_outputs=2)

        tf.identity(error_rate[0], "error_rate")
        tf.summary.scalar("error_rate", error_rate[1])
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
                eval_metric_ops=dict(error_rate=error_rate)
            )
