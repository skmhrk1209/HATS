import tensorflow as tf
import numpy as np
import os
import metrics
from algorithms import *


class Model(object):

    class AccuracyType:
        FULL_SEQUENCE, EDIT_DISTANCE = range(2)

    def __init__(self, convolutional_network, attention_network,
                 num_classes, data_format, accuracy_type, hyper_params):

        self.convolutional_network = convolutional_network
        self.attention_network = attention_network
        self.num_classes = num_classes
        self.data_format = data_format
        self.accuracy_type = accuracy_type
        self.hyper_params = hyper_params

    def __call__(self, features, labels, mode):

        images = features["image"]

        feature_maps = self.convolutional_network(
            inputs=images,
            training=mode == tf.estimator.ModeKeys.TRAIN
        )

        def flatten_images(inputs, data_format):

            input_shape = inputs.get_shape().as_list()
            output_shape = ([-1, input_shape[1], np.prod(input_shape[2:4])] if self.data_format == "channels_first" else
                            [-1, np.prod(input_shape[1:3]), input_shape[3]])

            return tf.reshape(inputs, output_shape)

        feature_vectors = flatten_images(feature_maps)

        lstm_cell = tf.nn.rnn_cell.LSTMCell(
            num_units=num_units,
            use_peepholes=True
        )

        attention_mechanism = tf.contrib.seq2seq.BahdanauAttention(
            num_units=num_units,
            memory=feature_vectors
        )

        ''' Perform a step of attention-wrapped RNN.
        - Step 1: Mix the `inputs` and previous step's `attention` output via `cell_input_fn`.
        - Step 2: Call the wrapped `cell` with this input and its previous state.
        - Step 3: Score the cell's output with `attention_mechanism`.
        - Step 4: Calculate the alignments by passing the score through the `normalizer`.
        - Step 5: Calculate the context vector as the inner product 
                  between the alignments and the attention_mechanism's values (memory).
        - Step 6: Calculate the attention output by concatenating the cell output and context 
                  through the attention layer (a linear layer with `attention_layer_size` outputs).
        '''
        attention_lstm_cell = tf.contrib.seq2seq.AttentionWrapper(
            cell=lstm_cell,
            attention_mechanism=attention_mechanism,
            cell_input_fn=lambda inputs, attention: f.layers.dense(
                inputs=tf.concat([inputs, attention[:, num_units:]], axis=-1),
                units=num_units
            ),
            output_attention=True,
            attention_layer=lambda inputs: inputs
        )

        sequence_length = tf.map_fn(
            fn=lambda is_not_blank: tf.count_nonzero(is_not_blank),
            elems=tf.not_equal(labels, self.num_classes - 1)
        )

        batch_size = tf.shape(sequence_length)[0]
        start_tokens = tf.constant(-1, shape=[batch_size])

        if mode == tf.estimator.ModeKeys.TRAIN:

            helper = tf.contrib.seq2seq.TrainingHelper(
                inputs=tf.one_hot(
                    indices=tf.concat([tf.expand_dims(start_tokens, axis=1), labels], axis=1),
                    depth=self.num_classes
                ),
                sequence_length=sequence_length,
                time_major=False
            )

        else:

            helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(
                embedding=lambda labels: tf.one_hot(
                    indices=labels,
                    depth=self.num_classes
                ),
                start_tokens=start_tokens,
                end_token=self.num_classes - 1
            )

        decoder = tf.contrib.seq2seq.BasicDecoder(
            cell=attention_lstm_cell,
            helper=helper,
            initial_state=attention_lstm_cell.zero_state(batch_size, tf.float32),
            output_layer=lambda inputs: tf.layers.dense(
                inputs=inputs,
                units=self.num_classes
            )
        )

        logits = tf.contrib.seq2seq.dynamic_decode(
            decoder=decoder,
            output_time_major=False,
            impute_finished=True,
            maximum_iterations=None,
            parallel_iterations=os.cpu_count(),
            swap_memory=False,
        )[0].rnn_output

        if mode == tf.estimator.ModeKeys.PREDICT:

            predictions = map_innermost_element(
                function=lambda logits: tf.argmax(
                    input=logits,
                    axis=1
                ),
                sequence=logits
            )

            while isinstance(merged_attention_maps, list):

                merged_attention_maps = map_innermost_list(
                    function=lambda merged_attention_maps: tf.stack(merged_attention_maps, axis=1),
                    sequence=merged_attention_maps
                )

            while isinstance(predictions, list):

                predictions = map_innermost_list(
                    function=lambda predictions: tf.stack(predictions, axis=1),
                    sequence=predictions
                )

            return tf.estimator.EstimatorSpec(
                mode=mode,
                predictions=dict(
                    images=images,
                    merged_attention_maps=merged_attention_maps,
                    predictions=predictions
                )
            )

        while all(flatten_innermost_element(map_innermost_element(lambda labels: len(labels.shape) > 1, labels))):

            labels = map_innermost_element(
                function=lambda labels: tf.unstack(labels, axis=1),
                sequence=labels
            )

        cross_entropy_losses = map_innermost_element(
            function=lambda logits_labels: tf.losses.sparse_softmax_cross_entropy(
                logits=logits_labels[0],
                labels=logits_labels[1]
            ),
            sequence=zip_innermost_element(logits, labels)
        )

        attention_map_losses = map_innermost_element(
            function=lambda attention_maps: tf.reduce_mean(
                tf.reduce_sum(tf.abs(attention_maps), axis=[1, 2, 3])
            ),
            sequence=attention_maps
        )

        losses = map_innermost_element(
            function=lambda cross_entropy_loss_attention_map_loss: (
                cross_entropy_loss_attention_map_loss[0] * self.hyper_params.cross_entropy_decay +
                cross_entropy_loss_attention_map_loss[1] * self.hyper_params.attention_map_decay
            ),
            sequence=zip_innermost_element(cross_entropy_losses, attention_map_losses)
        )

        loss = tf.reduce_mean(losses)

        # ==========================================================================================
        tf.summary.image("images", images, max_outputs=2)

        map_innermost_element(
            function=lambda indices_merged_attention_maps: tf.summary.image(
                name="merged_attention_maps_{}".format("_".join(map(str, indices_merged_attention_maps[0]))),
                tensor=indices_merged_attention_maps[1],
                max_outputs=2
            ),
            sequence=enumerate_innermost_element(merged_attention_maps)
        )

        map_innermost_element(
            function=lambda indices_cross_entropy_loss: tf.summary.scalar(
                name="cross_entropy_loss_{}".format("_".join(map(str, indices_cross_entropy_loss[0]))),
                tensor=indices_cross_entropy_loss[1]
            ),
            sequence=enumerate_innermost_element(cross_entropy_losses)
        )

        map_innermost_element(
            function=lambda indices_attention_map_loss: tf.summary.scalar(
                name="attention_map_loss_{}".format("_".join(map(str, indices_attention_map_loss[0]))),
                tensor=indices_attention_map_loss[1]
            ),
            sequence=enumerate_innermost_element(attention_map_losses)
        )

        map_innermost_element(
            function=lambda indices_loss: tf.summary.scalar(
                name="loss_{}".format("_".join(map(str, indices_loss[0]))),
                tensor=indices_loss[1]
            ),
            sequence=enumerate_innermost_element(losses)
        )
        # ==========================================================================================

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
