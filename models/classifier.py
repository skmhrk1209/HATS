import tensorflow as tf
import numpy as np
import metrics
from algorithms import *
from networks import ops


class Classifier(object):

    def __init__(self, backbone_network, num_classes, data_format, hyper_params):

        self.backbone_network = backbone_network
        self.num_classes = num_classes
        self.data_format = data_format
        self.hyper_params = hyper_params

    def __call__(self, images, labels, mode):

        feature_maps = self.backbone_network(
            inputs=images,
            training=mode == tf.estimator.ModeKeys.TRAIN
        )

        print(feature_maps)

        feature_vectors = ops.global_average_pooling2d(
            inputs=feature_maps[-1],
            data_format=self.data_format
        )

        logits = tf.layers.dense(
            inputs=feature_vectors,
            units=self.num_classes
        )

        predictions = tf.argmax(logits, axis=-1)

        if mode == tf.estimator.ModeKeys.PREDICT:

            return tf.estimator.EstimatorSpec(
                mode=mode,
                predictions=dict(
                    images=images,
                    predictions=predictions
                )
            )

        loss = tf.losses.sparse_softmax_cross_entropy(
            labels=labels,
            logits=logits
        )

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

            accuracy = tf.metrics.accuracy(
                labels=labels,
                predictions=predictions
            )

            return tf.estimator.EstimatorSpec(
                mode=mode,
                loss=loss,
                eval_metric_ops=dict(accuracy=accuracy)
            )
