import tensorflow as tf
import numpy as np
from algorithms.sequence import *


class ACNN(object):

    def __init__(self, convolutional_network, attention_network,
                 num_classes, data_format, hyper_params):

        self.convolutional_network = convolutional_network
        self.attention_network = attention_network
        self.num_classes = num_classes
        self.data_format = data_format
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

        merged_attention_maps = map_innermost(
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

        feature_vectors = map_innermost(
            function=lambda attention_maps: tf.layers.flatten(tf.matmul(
                a=flatten_images(feature_maps, self.data_format),
                b=flatten_images(attention_maps, self.data_format),
                transpose_a=False if self.data_format == "channels_first" else True,
                transpose_b=True if self.data_format == "channels_first" else False
            )),
            sequence=attention_maps
        )

        logits = map_innermost(
            function=lambda feature_vectors: tf.layers.dense(
                inputs=feature_vectors,
                units=self.num_classes,
                name="logits",
                reuse=tf.AUTO_REUSE
            ),
            sequence=feature_vectors
        )

        predictions = map_innermost(
            function=lambda logits: tf.argmax(
                input=logits,
                axis=-1,
                output_type=tf.int32
            ),
            sequence=logits
        )

        while all_innermost(map_innermost(lambda labels: len(labels.shape) > 1, labels)):

            labels = map_innermost(
                function=lambda labels: tf.unstack(labels, axis=1),
                sequence=labels
            )

        if mode == tf.estimator.ModeKeys.PREDICT:

            return tf.estimator.EstimatorSpec(
                mode=mode,
                predictions={}
            )

        cross_entropy_losses = map_innermost(
            function=lambda labels_and_logits: tf.losses.sparse_softmax_cross_entropy(
                labels=labels_and_logits[0],
                logits=labels_and_logits[1]
            ),
            sequence=zip_innermost(labels, logits)
        )

        attention_map_losses = map_innermost(
            function=lambda attention_maps: tf.reduce_mean(
                tf.reduce_sum(tf.abs(attention_maps), axis=[1, 2, 3])
            ),
            sequence=attention_maps
        )

        '''
        total_variation_losses = map_innermost(
            function=lambda attention_maps: tf.reduce_mean(
                tf.image.total_variation(attention_maps)
            ),
            sequence=attention_maps
        )
        '''

        losses = map_innermost(
            function=lambda cross_entropy_loss_and_attention_map_loss: (
                cross_entropy_loss_and_attention_map_loss[0] * self.hyper_params.cross_entropy_decay +
                cross_entropy_loss_and_attention_map_loss[1] * self.hyper_params.attention_map_decay
            ),
            sequence=zip_innermost(cross_entropy_losses, attention_map_losses)
        )

        enumerate_map_innermost(
            function=lambda indices, loss: tf.identity(
                input=loss,
                name="loss_{}".format("_".join([str(index) for index in indices]))
            ),
            sequence=losses
        )

        accuracies = map_innermost(
            function=lambda labels_and_predictions: tf.metrics.accuracy(
                labels=labels_and_predictions[0],
                predictions=labels_and_predictions[1]
            ),
            sequence=zip_innermost(labels, predictions)
        )

        enumerate_map_innermost(
            function=lambda indices, accuracy: tf.identity(
                input=accuracy[0],
                name="accuracy_{}".format("_".join([str(index) for index in indices]))
            ),
            sequence=accuracies
        )

        # ==========================================================================================
        tf.summary.image("images", images, max_outputs=2)

        for variable in tf.trainable_variables("attention_network"):
            tf.summary.histogram(variable.name, variable)

        enumerate_map_innermost(
            function=lambda indices, merged_attention_maps: tf.summary.image(
                name="merged_attention_maps_{}".format("_".join([str(index) for index in indices])),
                tensor=merged_attention_maps,
                max_outputs=2
            ),
            sequence=merged_attention_maps
        )

        enumerate_map_innermost(
            function=lambda indices, cross_entropy_loss: tf.summary.scalar(
                name="cross_entropy_loss_{}".format("_".join([str(index) for index in indices])),
                tensor=cross_entropy_loss
            ),
            sequence=cross_entropy_losses
        )

        enumerate_map_innermost(
            function=lambda indices, attention_map_loss: tf.summary.scalar(
                name="attention_map_loss_{}".format("_".join([str(index) for index in indices])),
                tensor=attention_map_loss
            ),
            sequence=attention_map_losses
        )

        enumerate_map_innermost(
            function=lambda indices, loss: tf.summary.scalar(
                name="loss_{}".format("_".join([str(index) for index in indices])),
                tensor=loss
            ),
            sequence=losses
        )

        enumerate_map_innermost(
            function=lambda indices, accuracy: tf.summary.scalar(
                name="accuracy_{}".format("_".join([str(index) for index in indices])),
                tensor=accuracy[1]
            ),
            sequence=accuracies
        )
        # ==========================================================================================

        loss = tf.reduce_mean(losses)

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
                eval_metric_ops={"accuracy": accuracy}
            )
