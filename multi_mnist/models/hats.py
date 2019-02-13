import tensorflow as tf
import numpy as np
import functools
import metrics
import summary
from networks import ops
from algorithms import *


class HATS(object):
    """ HATS: Hierarchical Attention-based Text Spotter """

    def __init__(self, backbone_network, attention_network,
                 num_units, num_classes, data_format, hyper_params):

        self.backbone_network = backbone_network
        self.attention_network = attention_network
        self.num_units = num_units
        self.num_classes = num_classes
        self.data_format = data_format
        self.hyper_params = hyper_params
        self.blank = num_classes - 1

    def __call__(self, images, labels, mode):
        # =========================================================================================
        # feature mapを計算
        feature_maps = self.backbone_network(
            inputs=images,
            training=mode == tf.estimator.ModeKeys.TRAIN
        )
        # =========================================================================================
        # attention mapを計算
        # 文字構造がnested listとして出力される
        # nested listはalgorithmsモジュール全般で処理する
        # TODO: sequence_lengthsを渡して冗長な計算を除去
        # TODO: 若干混み合った計算が必要, 出来るだけ抽象的に描きたい
        attention_maps = self.attention_network(
            inputs=feature_maps,
            training=mode == tf.estimator.ModeKeys.TRAIN
        )
        # =========================================================================================
        # 空間方向にflattenするための便利関数

        def spatial_flatten(inputs, data_format):
            inputs_shape = inputs.shape.as_list()
            outputs_shape = ([-1, inputs_shape[1], np.prod(inputs_shape[2:])] if data_format == "channels_first" else
                             [-1, np.prod(inputs_shape[1:-1]), inputs_shape[-1]])
            return tf.reshape(inputs, outputs_shape)
        # attention mapによるfeature extraction
        feature_vectors = map_innermost_element(
            function=lambda attention_maps: tf.layers.flatten(tf.matmul(
                a=spatial_flatten(feature_maps, self.data_format),
                b=spatial_flatten(attention_maps, self.data_format),
                transpose_a=False if self.data_format == "channels_first" else True,
                transpose_b=True if self.data_format == "channels_first" else False
            )),
            sequence=attention_maps
        )
        # =========================================================================================
        # logitの前に何層かFCを入れておく
        # TODO: 本当に必要?
        for i, num_units in enumerate(self.num_units):

            with tf.variable_scope("dense_block_{}".format(i)):

                feature_vectors = map_innermost_element(
                    function=compose(
                        lambda inputs: tf.layers.dense(
                            inputs=inputs,
                            units=num_units,
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
        # =========================================================================================
        # logitのinitializationは特に重要ではない?
        # softmaxだからとりあえずxavier initialization
        logits = map_innermost_element(
            function=lambda feature_vectors: tf.layers.dense(
                inputs=feature_vectors,
                units=self.num_classes,
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
        # argmaxで文字予測
        predictions = map_innermost_element(
            function=lambda logits: tf.argmax(
                input=logits,
                axis=-1,
                output_type=tf.int32
            ),
            sequence=logits
        )
        # =========================================================================================
        # attention mapは可視化のためにチャンネルをマージする
        attention_maps = map_innermost_element(
            function=lambda attention_maps: tf.reduce_sum(
                input_tensor=attention_maps,
                axis=1 if self.data_format == "channels_first" else 3,
                keepdims=True
            ),
            sequence=attention_maps
        )
        # =========================================================================================
        # prediction mode
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
        # =========================================================================================
        # logits, predictions同様にlabelsもunstackしてnested listにしておく
        # おそらくtf.reshape([-1, labels.shape[-1]])でも同様だが少し怖い
        while all(flatten_innermost_element(map_innermost_element(lambda labels: len(labels.shape) > 1, labels))):
            labels = map_innermost_element(
                function=lambda labels: tf.unstack(labels, axis=1),
                sequence=labels
            )
        # =========================================================================================
        # 簡単のため，単語構造のみを残して残りはバッチ方向に展開
        # [batch_size, max_sequence_length_0, ..., max_equence_length_N, ...] =>
        # [batch_size * max_sequence_length_0 * ..., max_equence_length_N, ...]
        labels = tf.concat(flatten_innermost_element(map_innermost_list(
            function=lambda labels: tf.stack(labels, axis=1),
            sequence=labels
        )), axis=0)
        logits = tf.concat(flatten_innermost_element(map_innermost_list(
            function=lambda logits: tf.stack(logits, axis=1),
            sequence=logits
        )), axis=0)
        predictions = tf.concat(flatten_innermost_element(map_innermost_list(
            function=lambda predictions: tf.stack(predictions, axis=1),
            sequence=predictions
        )), axis=0)
        # =========================================================================================
        # blankのみ含む単語(つまり存在しない)を削除
        indices = tf.where(tf.reduce_any(tf.not_equal(labels, self.blank), axis=1))
        labels = tf.gather_nd(labels, indices)
        logits = tf.gather_nd(logits, indices)
        # =========================================================================================
        # lossがblankを含まないようにマスク
        sequence_lengths = tf.count_nonzero(tf.not_equal(labels, self.blank), axis=1)
        # 最初のblankはEOSとして残しておく
        sequence_lengths += tf.ones_like(sequence_lengths)
        # binary mask
        sequence_mask = tf.sequence_mask(sequence_lengths, labels.shape[-1], dtype=tf.int32)
        # =========================================================================================
        # cross entropy loss
        loss = tf.contrib.seq2seq.sequence_loss(
            logits=logits,
            targets=labels,
            weights=tf.cast(sequence_mask, tf.float32),
            average_across_timesteps=True,
            average_across_batch=True
        )
        # =========================================================================================
        # 余分なblankを除去した単語の正解率を求める
        word_accuracy = tf.reduce_mean(tf.cast(tf.reduce_all(tf.equal(
            x=predictions * sequence_mask,
            y=labels * sequence_mask
        ), axis=1), dtype=tf.float32), name="word_accuracy")
        # =========================================================================================
        # TODO: なぜかedit distanceのshapeがloggingの際に異なる
        # =========================================================================================
        # tensorboard用のsummary
        summary.scalar(word_accuracy, name="word_accuracy")
        summary.image(images, name="images", data_format=self.data_format, max_outputs=2)
        for indices, attention_maps in flatten_innermost_element(enumerate_innermost_element(attention_maps)):
            summary.image(attention_maps, name="attention_maps_{}".format("_".join(map(str, indices))), data_format=self.data_format, max_outputs=2)
        # =========================================================================================
        # training mode
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
        # =========================================================================================
        # evaluation mode
        if mode == tf.estimator.ModeKeys.EVAL:

            return tf.estimator.EstimatorSpec(
                mode=mode,
                loss=loss,
                eval_metric_ops=dict(
                    word_accuracy=tf.metrics.mean(word_accuracy)
                )
            )
        # =========================================================================================
