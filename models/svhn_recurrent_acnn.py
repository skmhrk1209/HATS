import tensorflow as tf
import numpy as np


class Model(object):

    def __init__(self, convolutional_network, recurrent_attention_network,
                 num_classes, num_digits, data_format, hyper_params):

        self.convolutional_network = convolutional_network
        self.recurrent_attention_network = recurrent_attention_network
        self.num_classes = num_classes
        self.num_digits = num_digits
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

        attention_maps_sequence = self.recurrent_attention_network(
            inputs=feature_maps,
            training=mode == tf.estimator.ModeKeys.TRAIN
        )

        merged_attention_maps_sequence = [tf.reduce_sum(
            input_tensor=attention_maps,
            axis=1 if self.data_format == "channels_first" else 3,
            keep_dims=True
        ) for attention_maps in attention_maps_sequence]

        def flatten_images(inputs, data_format):

            input_shape = inputs.get_shape().as_list()
            output_shape = ([-1, input_shape[1], np.prod(input_shape[2:4])] if self.data_format == "channels_first" else
                            [-1, np.prod(input_shape[1:3]), input_shape[3]])

            return tf.reshape(inputs, output_shape)

        feature_vectors_sequence = [tf.matmul(
            a=flatten_images(feature_maps, self.data_format),
            b=flatten_images(attention_maps, self.data_format),
            transpose_a=False if self.data_format == "channels_first" else True,
            transpose_b=True if self.data_format == "channels_first" else False
        ) for attention_maps in attention_maps_sequence]

        feature_vectors_sequence = [tf.layers.flatten(
            inputs=feature_vectors
        ) for feature_vectors in feature_vectors_sequence]

        logits_sequence = [tf.layers.dense(
            inputs=feature_vectors,
            units=self.num_classes,
            kernel_initializer=tf.variance_scaling_initializer(
                scale=2.0,
                mode="fan_in",
                distribution="normal",
            ),
            bias_initializer=tf.zeros_initializer(),
            name="multi_logits",
            reuse=tf.AUTO_REUSE
        ) for feature_vectors in feature_vectors_sequence]

        labels_sequence = tf.unstack(labels, axis=1)

        classes_sequence = [tf.argmax(
            input=logits,
            axis=-1,
            name="classes"
        ) for logits in logits_sequence]

        cross_entropy_loss_sequence = [tf.losses.sparse_softmax_cross_entropy(
            labels=labels,
            logits=logits
        ) for labels, logits in zip(labels_sequence, logits_sequence)]

        total_cross_entropy_loss = tf.reduce_mean(cross_entropy_loss_sequence)

        attention_map_loss_sequence = [tf.reduce_mean(tf.reduce_sum(
            input_tensor=tf.abs(attention_maps),
            axis=[1, 2, 3]
        )) for attention_maps in attention_maps_sequence]

        total_attention_map_loss = tf.reduce_mean(attention_map_loss_sequence)

        total_variation_loss_sequence = [tf.reduce_mean(
            input_tensor=tf.image.total_variation(attention_maps)
        ) for attention_maps in attention_maps_sequence]

        total_total_variation_loss = tf.reduce_mean(total_variation_loss_sequence)

        loss_sequence = [(
            total_cross_entropy_loss * self.hyper_params.cross_entropy_decay +
            total_attention_map_loss * self.hyper_params.attention_map_decay +
            total_total_variation_loss * self.hyper_params.total_variation_decay
        ) for total_cross_entropy_loss, total_attention_map_loss, total_total_variation_loss in zip(
            cross_entropy_loss_sequence, attention_map_loss_sequence, total_variation_loss_sequence
        )]

        total_loss = tf.reduce_mean(loss_sequence)

        accuracy_sequence = [tf.metrics.accuracy(
            labels=labels,
            predictions=classes
        ) for labels, classes in zip(labels_sequence, classes_sequence)]

        total_accuracy = (
            tf.reduce_mean([accuracy[0] for accuracy in accuracy_sequence]),
            tf.group(*[accuracy[1] for accuracy in accuracy_sequence])
        )

        # ==========================================================================================
        tf.summary.image(
            name="features",
            tensor=features,
            max_outputs=8
        )

        [tf.summary.image(
            name="merged_attention_maps_sequence_{}".format(i),
            tensor=merged_attention_maps,
            max_outputs=8
        ) for i, merged_attention_maps in enumerate(merged_attention_maps_sequence)]

        [tf.summary.scalar("cross_entropy_loss_sequence_{}".format(i), total_cross_entropy_loss)
         for i, total_cross_entropy_loss in enumerate(cross_entropy_loss_sequence)]
        tf.summary.scalar("total_cross_entropy_loss", total_cross_entropy_loss)

        [tf.summary.scalar("attention_map_loss_sequence_{}".format(i), total_attention_map_loss)
         for i, total_attention_map_loss in enumerate(attention_map_loss_sequence)]
        tf.summary.scalar("total_attention_map_loss", total_attention_map_loss)

        [tf.summary.scalar("total_variation_loss_sequence_{}".format(i), total_total_variation_loss)
         for i, total_total_variation_loss in enumerate(total_variation_loss_sequence)]
        tf.summary.scalar("total_total_variation_loss", total_total_variation_loss)

        [tf.summary.scalar("loss_sequence_{}".format(i), total_loss)
         for i, total_loss in enumerate(loss_sequence)]
        tf.summary.scalar("total_loss", total_loss)

        [tf.summary.scalar("accuracy_sequence_{}".format(i), accuracy[0])
         for i, accuracy in enumerate(accuracy_sequence)]
        tf.summary.scalar("total_accuracy", total_accuracy[0])
        # ==========================================================================================

        if mode == tf.estimator.ModeKeys.TRAIN:

            with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):

                global_step = tf.train.get_or_create_global_step()

                optimizer = tf.train.AdamOptimizer()

                train_op = optimizer.minimize(
                    loss=total_loss,
                    global_step=global_step
                )

            return tf.estimator.EstimatorSpec(
                mode=mode,
                loss=total_loss,
                train_op=train_op
            )

        if mode == tf.estimator.ModeKeys.EVAL:

            return tf.estimator.EstimatorSpec(
                mode=mode,
                loss=total_loss,
                eval_metric_ops={"total_accuracy": total_accuracy}
            )
