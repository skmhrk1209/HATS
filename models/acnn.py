import tensorflow as tf
import numpy as np
import sys

sys.path.append("../utils")
from utils import attr_dict


class Model(object):

    def __init__(self, convolutional_network, attention_network):

        self.convolutional_network = convolutional_network
        self.attention_network = attention_network

    def __call__(self, features, labels, mode, params):

        params = attr_dict.AttrDict(params)
        predictions = attr_dict.AttrDict()

        images = features["image"]

        predictions.images = images

        tf.summary.image(
            name="images",
            tensor=images,
            max_outputs=10
        )

        feature_maps = self.convolutional_network(
            inputs=images
        )

        predictions.feature_maps = feature_maps

        attention_maps = self.attention_network(
            inputs=feature_maps,
            training=mode == tf.estimator.ModeKeys.TRAIN
        )

        predictions.attention_maps = attention_maps

        tf.summary.image(
            name="attention_maps",
            tensor=tf.reduce_sum(
                input_tensor=attention_maps,
                axis=3,
                keep_dims=True
            ),
            max_outputs=10
        )

        shape = feature_maps.shape.as_list()
        feature_maps = tf.reshape(
            tensor=feature_maps,
            shape=[-1, np.prod(shape[1:3]), shape[3]]
        )

        shape = attention_maps.shape.as_list()
        attention_maps = tf.reshape(
            tensor=attention_maps,
            shape=[-1, np.prod(shape[1:3]), shape[3]]
        )

        feature_vectors = tf.matmul(
            a=feature_maps,
            b=attention_maps,
            transpose_a=True,
            transpose_b=False
        )

        feature_vectors = tf.layers.flatten(feature_vectors)

        feature_vectors = tf.layers.dense(
            inputs=feature_vectors,
            units=1024
        )

        feature_vectors = tf.nn.relu(feature_vectors)

        logits = tf.layers.dense(
            inputs=feature_vectors,
            units=1000
        )

        predictions.classes = tf.argmax(
            input=logits,
            axis=-1
        ),
        predictions.softmax = tf.nn.softmax(
            logits=logits,
            dim=-1,
            name="softmax"
        )

        if mode == tf.estimator.ModeKeys.PREDICT:

            return tf.estimator.EstimatorSpec(
                mode=mode,
                predictions=predictions
            )

        loss = tf.losses.sparse_softmax_cross_entropy(
            labels=labels,
            logits=logits
        )

        loss += tf.reduce_mean(tf.reduce_sum(
            input_tensor=tf.abs(attention_maps),
            axis=[1, 2]
        )) * params.attention_decay

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

            eval_metric_ops = dict(
                accuracy=tf.metrics.accuracy(
                    labels=labels,
                    predictions=predictions.classes
                )
            )

            return tf.estimator.EstimatorSpec(
                mode=mode,
                loss=loss,
                eval_metric_ops=eval_metric_ops
            )
