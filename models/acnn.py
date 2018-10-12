import tensorflow as tf
import numpy as np


class Model(object):

    def __init__(self, attention_network, convolutional_network,
                 num_classes, data_format, hyper_params):

        self.attention_network = attention_network
        self.convolutional_network = convolutional_network
        self.num_classes = num_classes
        self.data_format = data_format
        self.hyper_params = hyper_params

    def __call__(self, features, labels, mode):
        ''' model function for ACNN

            features:   batch of features from input_fn
            labels:     batch of labels from input_fn
            mode:       enum { TRAIN, EVAL, PREDICT }
        '''

        feature_maps = self.convolutional_network(
            inputs=features,
            training=mode == tf.estimator.ModeKeys.TRAIN
        )

        attention_maps = self.attention_network(
            inputs=feature_maps,
            training=mode == tf.estimator.ModeKeys.TRAIN
        )

        tf.summary.image(
            name="attention_map",
            tensor=tf.reduce_sum(
                input_tensor=attention_maps,
                axis=1 if self.data_format == "channels_first" else 3,
                keepdims=True
            )
        )

        shape = feature_maps.get_shape().as_list()
        feature_maps = tf.reshape(
            tensor=feature_maps,
            shape=([-1, shape[1], np.prod(shape[2:4])] if self.data_format == "channels_first" else
                   [-1, np.prod(shape[1:3]), shape[3]])
        )

        shape = attention_maps.get_shape().as_list()
        attention_maps = tf.reshape(
            tensor=attention_maps,
            shape=([-1, shape[1], np.prod(shape[2:4])] if self.data_format == "channels_first" else
                   [-1, np.prod(shape[1:3]), shape[3]])
        )

        feature_vectors = tf.matmul(
            a=feature_maps,
            b=attention_maps,
            transpose_a=False if self.data_format == "channels_first" else True,
            transpose_b=True if self.data_format == "channels_first" else False
        )

        feature_vectors = tf.layers.flatten(feature_vectors)

        logits = tf.layers.dense(
            inputs=feature_vectors,
            units=self.num_classes
        )

        print("num params: {}".format(
            np.sum([np.prod(variable.get_shape().as_list())
                    for variable in tf.global_variables()])
        ))

        softmax = tf.nn.softmax(logits, dim=-1, name="softmax")
        classes = tf.argmax(logits, axis=-1, name="classes")

        loss = tf.losses.sparse_softmax_cross_entropy(
            labels=labels,
            logits=logits
        )

        loss += tf.add_n([
            tf.nn.l2_loss(variable) for variable in tf.trainable_variables()
            if "batch_normalization" not in variable.name
        ]) * self.hyper_params.weight_decay

        loss += tf.reduce_mean(
            input_tensor=tf.reduce_mean(
                input_tensor=tf.reduce_sum(
                    input_tensor=tf.abs(attention_maps),
                    axis=2 if self.data_format == "channels_first" else 1
                ),
                axis=1
            ),
            axis=0
        ) * self.hyper_params.attention_decay

        tf.summary.scalar("loss", loss)

        accuracy = tf.metrics.accuracy(
            labels=labels,
            predictions=classes
        )

        accuracy_top_5 = tf.metrics.mean(
            tf.nn.in_top_k(
                predictions=logits,
                targets=labels,
                k=5
            )
        )

        tf.summary.scalar("accuracy", accuracy[1])
        tf.summary.scalar("accuracy_top_5", accuracy_top_5[1])

        if mode == tf.estimator.ModeKeys.TRAIN:

            with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):

                global_step = tf.train.get_or_create_global_step()

                optimizer = tf.train.MomentumOptimizer(
                    learning_rate=self.hyper_params.learning_rate_fn(global_step),
                    momentum=self.hyper_params.momentum
                )

                train_op = optimizer.minimize(
                    loss=loss,
                    global_step=global_step
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
                    "accuracy": accuracy,
                    "accuracy_top_5": accuracy_top_5
                }
            )
