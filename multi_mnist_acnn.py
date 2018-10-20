from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import cv2
import argparse
import itertools
import functools
import operator
import glob
import os

parser = argparse.ArgumentParser()
parser.add_argument("--steps", type=int, default=10000, help="number of training steps")
parser.add_argument("--num_epochs", type=int, default=100, help="number of training epochs")
parser.add_argument("--batch_size", type=int, default=100, help="batch size")
parser.add_argument("--model_dir", type=str, default="multi_mnist_acnn_model", help="model directory")
parser.add_argument('--train', action="store_true", help="with training")
parser.add_argument('--eval', action="store_true", help="with evaluation")
parser.add_argument('--predict', action="store_true", help="with prediction")
parser.add_argument('--gpu', type=str, default="0", help="gpu id")
args = parser.parse_args()

tf.logging.set_verbosity(tf.logging.INFO)


def scale(input, input_min, input_max, output_min, output_max):

    return output_min + (input - input_min) / (input_max - input_min) * (output_max - output_min)


def acnn_model_fn(features, labels, mode, params):
    """""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    model function for ACNN

    features:   batch of features from input_fn
    labels:     batch of labels from input_fn
    mode:       enum { TRAIN, EVAL, PREDICT }
    """""""""""""""""""""""""""""""""""""""""""""""""""""""""""

    """""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    convolutional layer 1
    (-1, 128, 128, 1) -> (-1, 64, 64, 32)
    """""""""""""""""""""""""""""""""""""""""""""""""""""""""""

    inputs = features["images"]

    inputs = tf.layers.conv2d(
        inputs=inputs,
        filters=32,
        kernel_size=3,
        strides=2,
        padding="same",
        activation=tf.nn.relu
    )

    """""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    convolutional layer 2
    (-1, 64, 64, 32) -> (-1, 32, 32, 64)
    """""""""""""""""""""""""""""""""""""""""""""""""""""""""""

    inputs = tf.layers.conv2d(
        inputs=inputs,
        filters=64,
        kernel_size=3,
        strides=2,
        padding="same",
        activation=tf.nn.relu
    )

    """""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    attention convolutional layer 1
    (-1, 32, 32, 64) -> (-1, 16, 16, 3)
    """""""""""""""""""""""""""""""""""""""""""""""""""""""""""

    attentions_sequence = [inputs] * 2

    attentions_sequence = [
        tf.layers.conv2d(
            inputs=attentions,
            filters=3,
            kernel_size=9,
            strides=2,
            padding="same",
            activation=tf.nn.relu,
            name="conv2d_0",
            reuse=tf.AUTO_REUSE
        ) for attentions in attentions_sequence
    ]

    """""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    attention convolutional layer 2
    (-1, 16, 16, 3) -> (-1, 8, 8, 3)
    """""""""""""""""""""""""""""""""""""""""""""""""""""""""""

    attentions_sequence = [
        tf.layers.conv2d(
            inputs=attentions,
            filters=3,
            kernel_size=9,
            strides=2,
            padding="same",
            activation=tf.nn.relu,
            name="conv2d_1",
            reuse=tf.AUTO_REUSE
        ) for attentions in attentions_sequence
    ]

    """""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    attention dense layer 3
    (-1, 8, 8, 3) -> (-1, 64)
    """""""""""""""""""""""""""""""""""""""""""""""""""""""""""

    shape = attentions.get_shape().as_list()

    attentions_sequence = [
        tf.layers.flatten(attentions)
        for attentions in attentions_sequence
    ]

    lstm_cell = tf.nn.rnn_cell.LSTMCell(
        num_units=64,
        use_peepholes=True,
        initializer=tf.variance_scaling_initializer(
            scale=2.0,
            mode="fan_in",
            distribution="normal",
        )
    )

    attentions_sequence, _ = tf.nn.static_rnn(
        cell=lstm_cell,
        inputs=attentions_sequence,
        initial_state=lstm_cell.zero_state(
            batch_size=tf.shape(inputs)[0],
            dtype=tf.float32
        )
    )

    """""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    attention dense layer 4
    (-1, 10) -> (-1, 8, 8, 3)
    """""""""""""""""""""""""""""""""""""""""""""""""""""""""""

    attentions_sequence = [
        tf.layers.dense(
            inputs=attentions,
            units=functools.reduce(operator.mul, shape[1:]),
            activation=tf.nn.relu,
            name="dense_0",
            reuse=tf.AUTO_REUSE
        ) for attentions in attentions_sequence
    ]

    attentions_sequence = [
        tf.reshape(
            tensor=attentions,
            shape=[-1] + shape[1:]
        ) for attentions in attentions_sequence
    ]

    """""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    attention deconvolutional layer 5
    (-1, 8, 8, 3) -> (-1, 16, 16, 9)
    """""""""""""""""""""""""""""""""""""""""""""""""""""""""""

    attentions_sequence = [
        tf.layers.conv2d_transpose(
            inputs=attentions,
            filters=9,
            kernel_size=3,
            strides=2,
            padding="same",
            activation=tf.nn.relu,
            name="deconv2d_0",
            reuse=tf.AUTO_REUSE
        ) for attentions in attentions_sequence
    ]

    """""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    attention deconvolutional layer 6
    (-1, 16, 16, 9) -> (-1, 32, 32, 9)
    """""""""""""""""""""""""""""""""""""""""""""""""""""""""""

    attentions_sequence = [
        tf.layers.conv2d_transpose(
            inputs=attentions,
            filters=9,
            kernel_size=3,
            strides=2,
            padding="same",
            activation=tf.nn.sigmoid,
            name="deconv2d_1",
            reuse=tf.AUTO_REUSE
        ) for attentions in attentions_sequence
    ]

    merged_attentions_sequence = [
        tf.reduce_sum(
            input_tensor=attention_maps,
            axis=1 if self.data_format == "channels_first" else 3,
            keep_dims=True
        ) for attention_maps in attention_maps_sequence
    ]

    [tf.summary.image(
        name="merged_attentions_sequence_{}".format(i),
        tensor=merged_attentions,
        max_outputs=4
    ) for i, merged_attentions in enumerate(merged_attentions_sequence)]

    """""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    extract layer
    (-1, 32, 32, 64), (-1, 32, 32, 9) -> (-1, 64, 9)
    """""""""""""""""""""""""""""""""""""""""""""""""""""""""""

    shape = inputs.get_shape().as_list()

    inputs = tf.reshape(
        tensor=inputs,
        shape=[-1, np.prod(shape[1:3]), shape[3]]
    )

    shape_sequence = [
        attentions.get_shape().as_list()
        for attentions in attentions_sequence
    ]

    attentions_sequence = [
        tf.reshape(
            tensor=attentions,
            shape=[-1, np.prod(shape[1:3]), shape[3]]
        ) for attentions, shape in zip(attentions_sequence, shape_sequence)
    ]

    inputs_sequence = [
        tf.matmul(
            a=inputs,
            b=attentions,
            transpose_a=True,
            transpose_b=False
        ) for attentions in attentions_sequence
    ]

    """""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    dense layer 3
    (-1, 64, 9) -> (-1, 1024)
    """""""""""""""""""""""""""""""""""""""""""""""""""""""""""

    inputs_sequence = [
        tf.layers.flatten(inputs)
        for inputs in inputs_sequence
    ]

    inputs_sequence = [
        tf.layers.dense(
            inputs=inputs,
            units=1024,
            activation=tf.nn.relu,
            name="dense_1",
            reuse=tf.AUTO_REUSE
        ) for inputs in inputs_sequence
    ]

    """""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    logits layer
    (-1, 1024) -> (-1, 10)
    """""""""""""""""""""""""""""""""""""""""""""""""""""""""""

    logits_sequence = [
        tf.layers.dense(
            inputs=inputs,
            units=10,
            name="logits",
            reuse=tf.AUTO_REUSE
        ) for inputs in inputs_sequence
    ]

    labels_sequence = tf.unstack(labels, axis=1)

    loss = tf.reduce_mean([
        tf.losses.sparse_softmax_cross_entropy(
            labels=labels,
            logits=logits
        ) for labels, logits in zip(labels_sequence, logits_sequence)
    ])

    loss += tf.reduce_mean([
        tf.reduce_mean(
            input_tensor=tf.reduce_sum(
                input_tensor=tf.abs(attentions),
                axis=[1, 2]
            ),
            axis=None
        ) for attentions in attentions_sequence
    ]) * params["attention_decay"]

    tf.summary.scalar("loss", loss)

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


def main(unused_argv):

    def make_multi_mnist(images, labels):

        multi_images = []
        multi_labels = []

        for i in range(50000):

            indices = np.random.randint(low=0, high=len(images), size=2)

            offsets = [
                np.random.randint(low=0, high=100, size=2),
                np.random.randint(low=0, high=100, size=2)
            ]

            back = np.zeros([128, 128, 1], np.float32)

            for image, offset in zip(images[indices], sorted(offsets, key=lambda offset: (offset[0], offset[1]))):

                back[offset[0]:offset[0]+28, offset[1]:offset[1]+28, :] += image

            multi_images.append(back)
            multi_labels.append(labels[indices])

        return np.array(multi_images). np.array(multi_labels)

    mnist = tf.contrib.learn.datasets.load_dataset("mnist")
    train_images = mnist.train.images.astype(np.float32).reshape([-1, 28, 28, 1])
    train_labels = mnist.train.labels.astype(np.int32)
    train_images, train_labels = make_multi_mnist(train_images, train_labels)

    mnist_classifier = tf.estimator.Estimator(
        model_fn=acnn_model_fn,
        model_dir=args.model_dir,
        config=tf.estimator.RunConfig().replace(
            session_config=tf.ConfigProto(
                gpu_options=tf.GPUOptions(
                    visible_device_list=args.gpu,
                    allow_growth=True
                )
            )
        ),
        params={
            "attention_decay": 1e-3
        }
    )

    if args.train:

        train_input_fn = tf.estimator.inputs.numpy_input_fn(
            x={"images": train_images},
            y=train_labels,
            batch_size=args.batch_size,
            num_epochs=args.num_epochs,
            shuffle=True
        )

        logging_hook = tf.train.LoggingTensorHook(
            tensors={
                "probabilities": "probabilities"
            },
            every_n_iter=100
        )

        mnist_classifier.train(
            input_fn=train_input_fn,
            hooks=[logging_hook]
        )


if __name__ == "__main__":
    tf.app.run()
