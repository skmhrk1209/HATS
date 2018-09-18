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

    predictions = {}
    predictions.update(features)

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

    attentions = inputs

    attentions = tf.layers.conv2d(
        inputs=attentions,
        filters=3,
        kernel_size=9,
        strides=2,
        padding="same",
        activation=tf.nn.relu
    )

    """""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    attention convolutional layer 2
    (-1, 16, 16, 3) -> (-1, 8, 8, 3)
    """""""""""""""""""""""""""""""""""""""""""""""""""""""""""

    attentions = tf.layers.conv2d(
        inputs=attentions,
        filters=3,
        kernel_size=9,
        strides=2,
        padding="same",
        activation=tf.nn.relu
    )

    """""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    attention dense layer 3
    (-1, 8, 8, 3) -> (-1, 10)
    """""""""""""""""""""""""""""""""""""""""""""""""""""""""""

    shape = attentions.get_shape().as_list()

    attentions = tf.layers.flatten(attentions)

    attentions = tf.layers.dense(
        inputs=attentions,
        units=10,
        activation=tf.nn.relu
    )

    """""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    attention dense layer 4
    (-1, 10) -> (-1, 8, 8, 3)
    """""""""""""""""""""""""""""""""""""""""""""""""""""""""""

    attentions = tf.layers.dense(
        inputs=attentions,
        units=functools.reduce(operator.mul, shape[1:]),
        activation=tf.nn.relu
    )

    attentions = tf.reshape(
        tensor=attentions,
        shape=[-1] + shape[1:]
    )

    """""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    attention deconvolutional layer 5
    (-1, 8, 8, 3) -> (-1, 16, 16, 9)
    """""""""""""""""""""""""""""""""""""""""""""""""""""""""""

    attentions = tf.layers.conv2d_transpose(
        inputs=attentions,
        filters=9,
        kernel_size=3,
        strides=2,
        padding="same",
        activation=tf.nn.relu
    )

    """""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    attention deconvolutional layer 6
    (-1, 16, 16, 9) -> (-1, 32, 32, 9)
    """""""""""""""""""""""""""""""""""""""""""""""""""""""""""

    attentions = tf.layers.conv2d_transpose(
        inputs=attentions,
        filters=9,
        kernel_size=3,
        strides=2,
        padding="same",
        activation=tf.nn.sigmoid
    )

    predictions["attentions"] = attentions

    """""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    extract layer
    (-1, 32, 32, 64), (-1, 32, 32, 9) -> (-1, 64, 9)
    """""""""""""""""""""""""""""""""""""""""""""""""""""""""""

    shape = inputs.get_shape().as_list()

    inputs = tf.reshape(
        tensor=inputs,
        shape=[-1, functools.reduce(operator.mul, shape[1:3]), shape[3]]
    )

    shape = attentions.get_shape().as_list()

    attentions = tf.reshape(
        tensor=attentions,
        shape=[-1, functools.reduce(operator.mul, shape[1:3]), shape[3]]
    )

    inputs = tf.matmul(
        a=inputs,
        b=attentions,
        transpose_a=True,
        transpose_b=False
    )

    """""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    dense layer 3
    (-1, 64, 9) -> (-1, 128)
    """""""""""""""""""""""""""""""""""""""""""""""""""""""""""

    inputs = tf.layers.flatten(inputs)

    inputs = tf.layers.dense(
        inputs=inputs,
        units=128,
        activation=tf.nn.relu
    )

    """""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    logits layer
    (-1, 128) -> (-1, 10)
    """""""""""""""""""""""""""""""""""""""""""""""""""""""""""

    logits = tf.layers.dense(
        inputs=inputs,
        units=10
    )

    predictions.update({
        "classes": tf.argmax(
            input=logits,
            axis=-1
        ),
        "sigmoid": tf.nn.sigmoid(
            x=logits,
            name="sigmoid"
        )
    })

    if mode == tf.estimator.ModeKeys.PREDICT:

        return tf.estimator.EstimatorSpec(
            mode=mode,
            predictions=predictions
        )

    loss = tf.losses.sigmoid_cross_entropy(
        multi_class_labels=labels,
        logits=logits
    )

    loss += tf.reduce_mean(
        input_tensor=tf.reduce_sum(
            input_tensor=tf.abs(attentions),
            axis=[1, 2]
        ),
        axis=None
    ) * params["attention_decay"]

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

        eval_metric_ops = {
            "accuracy": tf.metrics.accuracy(
                labels=labels,
                predictions=predictions["classes"]
            )
        }

        return tf.estimator.EstimatorSpec(
            mode=mode,
            loss=loss,
            eval_metric_ops=eval_metric_ops
        )


def main(unused_argv):

    def random_resize_with_pad(image, size, mode, **kwargs):

        dy = size[0] - image.shape[0]
        dx = size[1] - image.shape[1]

        wy = np.random.randint(low=0, high=dy)
        wx = np.random.randint(low=0, high=dx)

        return np.pad(image, [[wy, dy - wy], [wx, dx - wx], [0, 0]], mode, **kwargs)

    def make_multi_mnist(images, labels, digits, size):

        multi_images = []
        multi_labels = []

        for _ in range(size):

            indices = np.random.randint(0, len(images), size=np.random.randint(1, digits + 1))

            multi_image = np.stack(images[indices], axis=3)
            multi_image = np.apply_along_axis(np.sum, axis=3, arr=multi_image)
            multi_image = np.clip(multi_image, 0, 1)
            multi_images.append(multi_image)

            multi_label = np.identity(10, dtype=np.int32)[labels[indices]]
            multi_label = np.any(multi_label, axis=0).astype(np.int32)
            multi_labels.append(multi_label)

        return np.array(multi_images), np.array(multi_labels)

    mnist = tf.contrib.learn.datasets.load_dataset("mnist")

    train_images = np.array([
        random_resize_with_pad(
            image=image.reshape([28, 28, 1]),
            size=[128, 128],
            mode="constant",
            constant_values=0
        ) for image in mnist.train.images
    ])

    eval_images = np.array([
        random_resize_with_pad(
            image=image.reshape([28, 28, 1]),
            size=[128, 128],
            mode="constant",
            constant_values=0
        ) for image in mnist.test.images
    ])

    train_labels = mnist.train.labels.astype(np.int32)
    eval_labels = mnist.test.labels.astype(np.int32)

    train_multi_images, train_multi_labels = make_multi_mnist(
        images=train_images,
        labels=train_labels,
        digits=4,
        size=train_images.shape[0]
    )

    eval_multi_images, eval_multi_labels = make_multi_mnist(
        images=eval_images,
        labels=eval_labels,
        digits=4,
        size=eval_images.shape[0]
    )

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
            "attention_decay": 1e-6
        }
    )

    if args.train:

        train_input_fn = tf.estimator.inputs.numpy_input_fn(
            x={"images": train_multi_images},
            y=train_multi_labels,
            batch_size=args.batch_size,
            num_epochs=args.num_epochs,
            shuffle=True
        )

        logging_hook = tf.train.LoggingTensorHook(
            tensors={
                "sigmoid": "sigmoid"
            },
            every_n_iter=100
        )

        mnist_classifier.train(
            input_fn=train_input_fn,
            hooks=[logging_hook]
        )

    if args.eval:

        eval_input_fn = tf.estimator.inputs.numpy_input_fn(
            x={"images": eval_multi_images},
            y=eval_multi_labels,
            batch_size=args.batch_size,
            num_epochs=1,
            shuffle=False
        )

        eval_results = mnist_classifier.evaluate(
            input_fn=eval_input_fn
        )

        print(eval_results)

    if args.predict:

        predict_input_fn = tf.estimator.inputs.numpy_input_fn(
            x={"images": eval_multi_images},
            y=eval_multi_labels,
            batch_size=args.batch_size,
            num_epochs=1,
            shuffle=False
        )

        predict_results = mnist_classifier.predict(
            input_fn=predict_input_fn
        )

        figure = plt.figure()
        artists = []

        for predict_result in itertools.islice(predict_results, 10):

            attention = predict_result["attentions"]
            attention = scale(attention, attention.min(), attention.max(), 0, 1)
            attention = np.apply_along_axis(np.sum, axis=-1, arr=attention)
            attention = cv2.resize(attention, (128, 128))

            image = predict_result["images"]
            image = image.repeat(3, axis=-1)
            image[:, :, 0] += attention

            artists.append([plt.imshow(image, animated=True)])

        anim = animation.ArtistAnimation(figure, artists, interval=1000, repeat=False)
        anim.save("mnist_attention.gif", writer="imagemagick")

        plt.show()


if __name__ == "__main__":
    tf.app.run()
