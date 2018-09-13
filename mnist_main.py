from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
import cv2
import argparse
import itertools
import functools
import operator
import glob
import utils

parser = argparse.ArgumentParser()
parser.add_argument("--num_steps", type=int, default=10000, help="number of training steps")
parser.add_argument("--num_epochs", type=int, default=100, help="number of training epochs")
parser.add_argument("--batch_size", type=int, default=100, help="batch size")
parser.add_argument("--model_dir", type=str, default="mnist_acnn_model", help="model directory")
parser.add_argument('--train', action="store_true", help="with training")
parser.add_argument('--eval', action="store_true", help="with evaluation")
parser.add_argument('--predict', action="store_true", help="with prediction")
parser.add_argument('--gpu', type=str, default="0", help="gpu id")
parser.add_argument('--data_format', type=str, default="channels_first", help="data_format")
args = parser.parse_args()

tf.logging.set_verbosity(tf.logging.INFO)


def acnn_model_fn(features, labels, mode, params, size, data_format):
    """""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    model function for ACNN

    features:   batch of features from input_fn
    labels:     batch of labels from input_fn
    mode:       enum { TRAIN, EVAL, REDICT }
    """""""""""""""""""""""""""""""""""""""""""""""""""""""""""

    inputs = features["images"]

    if data_format == "channels_first":

        inputs = tf.transpose(inputs, [0, 3, 1, 2])

    """""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    convolutional layer 1
    (-1, 64, 64, 1) -> (-1, 64, 64, 32)
    """""""""""""""""""""""""""""""""""""""""""""""""""""""""""

    inputs = tf.layers.conv2d(
        inputs=inputs,
        filters=32,
        kernel_size=5,
        strides=1,
        padding="same",
        data_format=data_format
    )
    '''
    inputs = utils.batch_normalization(data_format)(
        inputs=inputs,
        training=mode == tf.estimator.ModeKeys.TRAIN,
        fused=True
    )
    '''
    inputs = tf.nn.relu(inputs)

    """""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    convolutional layer 2
    (-1, 64, 64, 32) -> (-1, 64, 64, 64)
    """""""""""""""""""""""""""""""""""""""""""""""""""""""""""

    inputs = tf.layers.conv2d(
        inputs=inputs,
        filters=64,
        kernel_size=5,
        strides=1,
        padding="same",
        data_format=data_format
    )
    '''
    inputs = utils.batch_normalization(data_format)(
        inputs=inputs,
        training=mode == tf.estimator.ModeKeys.TRAIN,
        fused=True
    )
    '''
    inputs = tf.nn.relu(inputs)

    """""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    attention convolutional layer 1
    (-1, 64, 64, 64) -> (-1, 32, 32, 3)
    """""""""""""""""""""""""""""""""""""""""""""""""""""""""""

    attentions = tf.layers.conv2d(
        inputs=inputs,
        filters=3,
        kernel_size=9,
        strides=2,
        padding="same",
        data_format=data_format
    )
    '''
    attentions = utils.batch_normalization(data_format)(
        inputs=attentions,
        training=mode == tf.estimator.ModeKeys.TRAIN,
        fused=True
    )
    '''
    attentions = tf.nn.relu(attentions)

    """""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    attention convolutional layer 2
    (-1, 32, 32, 3) -> (-1, 16, 16, 3)
    """""""""""""""""""""""""""""""""""""""""""""""""""""""""""

    attentions = tf.layers.conv2d(
        inputs=attentions,
        filters=3,
        kernel_size=9,
        strides=2,
        padding="same",
        data_format=data_format
    )
    '''
    attentions = utils.batch_normalization(data_format)(
        inputs=attentions,
        training=mode == tf.estimator.ModeKeys.TRAIN,
        fused=True
    )
    '''
    attentions = tf.nn.relu(attentions)

    """""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    attention dense layer 3
    (-1, 16, 16, 3) -> (-1, 10)
    """""""""""""""""""""""""""""""""""""""""""""""""""""""""""

    shape = attentions.get_shape().as_list()

    attentions = tf.layers.flatten(attentions)

    attentions = tf.layers.dense(
        inputs=attentions,
        units=10
    )
    '''
    attentions = tf.layers.batch_normalization(
        inputs=attentions,
        training=mode == tf.estimator.ModeKeys.TRAIN,
        fused=True
    )
    '''
    attentions = tf.nn.relu(attentions)

    """""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    attention dense layer 4
    (-1, 10) -> (-1, 16, 16, 3)
    """""""""""""""""""""""""""""""""""""""""""""""""""""""""""

    attentions = tf.layers.dense(
        inputs=attentions,
        units=functools.reduce(operator.mul, shape[1:])
    )
    '''
    attentions = tf.layers.batch_normalization(
        inputs=attentions,
        training=mode == tf.estimator.ModeKeys.TRAIN,
        fused=True
    )
    '''
    attentions = tf.nn.relu(attentions)

    attentions = tf.reshape(
        tensor=attentions,
        shape=[-1] + shape[1:]
    )

    """""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    attention deconvolutional layer 5
    (-1, 16, 16, 3) -> (-1, 32, 32, 9)
    """""""""""""""""""""""""""""""""""""""""""""""""""""""""""

    attentions = tf.layers.conv2d_transpose(
        inputs=attentions,
        filters=9,
        kernel_size=3,
        strides=2,
        padding="same",
        data_format=data_format
    )
    '''
    attentions = utils.batch_normalization(data_format)(
        inputs=attentions,
        training=mode == tf.estimator.ModeKeys.TRAIN,
        fused=True
    )
    '''
    attentions = tf.nn.relu(attentions)

    """""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    attention deconvolutional layer 6
    (-1, 32, 32, 9) -> (-1, 64, 64, 9)
    """""""""""""""""""""""""""""""""""""""""""""""""""""""""""

    attentions = tf.layers.conv2d_transpose(
        inputs=attentions,
        filters=9,
        kernel_size=3,
        strides=2,
        padding="same",
        data_format=data_format
    )
    '''
    attentions = utils.batch_normalization(data_format)(
        inputs=attentions,
        training=mode == tf.estimator.ModeKeys.TRAIN,
        fused=True
    )
    '''
    attentions = tf.nn.sigmoid(attentions)

    """""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    extract layer
    (-1, 64, 64, 64), (-1, 64, 64, 9) -> (-1, 64, 9)
    """""""""""""""""""""""""""""""""""""""""""""""""""""""""""

    inputs = utils.flatten_images(inputs, data_format)

    attentions = utils.flatten_images(attentions, data_format)

    inputs = tf.matmul(
        a=inputs,
        b=attentions,
        transpose_a=False if data_format == "channels_first" else True,
        transpose_b=True if data_format == "channels_first" else False
    )

    """""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    dense layer 1
    (-1, 512, 9) -> (-1, 1024)
    """""""""""""""""""""""""""""""""""""""""""""""""""""""""""

    inputs = tf.layers.flatten(inputs)

    inputs = tf.layers.dense(
        inputs=inputs,
        units=1024
    )
    '''
    inputs = tf.layers.batch_normalization(
        inputs=inputs,
        training=mode == tf.estimator.ModeKeys.TRAIN,
        fused=True
    )
    '''
    inputs = tf.nn.relu(inputs)

    inputs = tf.layers.dropout(
        inputs=inputs,
        rate=0.4,
        training=mode == tf.estimator.ModeKeys.TRAIN
    )

    """""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    logits layer
    (-1, 1024) -> (-1, 10)
    """""""""""""""""""""""""""""""""""""""""""""""""""""""""""

    logits = tf.layers.dense(
        inputs=inputs,
        units=10
    )

    attentions = utils.chunk_images(
        inputs=attentions,
        size=size,
        data_format=data_format
    )

    if data_format == "channels_first":

        attentions = tf.transpose(attentions, [0, 2, 3, 1])

    predictions = {
        "classes": tf.argmax(
            input=logits,
            axis=1
        ),
        "probabilities": tf.nn.softmax(
            logits=logits,
            name="softmax"
        ),
        "attentions": attentions
    }

    predictions.update(features)

    if mode == tf.estimator.ModeKeys.PREDICT:

        return tf.estimator.EstimatorSpec(
            mode=mode,
            predictions=predictions
        )

    loss = tf.losses.sparse_softmax_cross_entropy(
        labels=labels,
        logits=logits
    )

    loss += tf.reduce_sum(tf.abs(attentions)) * params["attention_decay"]

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

    def resize_with_pad(image, size):

        diff_y = size[0] - image.shape[0]
        diff_x = size[1] - image.shape[1]

        pad_width_y = np.random.randint(low=0, high=diff_y)
        pad_width_x = np.random.randint(low=0, high=diff_x)

        return np.pad(image, [[pad_width_y, diff_y - pad_width_y], [pad_width_x, diff_x - pad_width_x], [0, 0]], "constant")

    mnist = tf.contrib.learn.datasets.load_dataset("mnist")
    train_images = np.array([resize_with_pad(image.reshape([28, 28, 1]), size=[64, 64])
                             for image in mnist.train.images])
    eval_images = np.array([resize_with_pad(image.reshape([28, 28, 1]), size=[64, 64])
                            for image in mnist.test.images])
    train_labels = mnist.train.labels.astype(np.int32)
    eval_labels = mnist.test.labels.astype(np.int32)

    mnist_classifier = tf.estimator.Estimator(
        model_fn=functools.partial(
            acnn_model_fn,
            size=[64, 64],
            data_format=args.data_format
        ),
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
            x={"images": train_images},
            y=train_labels,
            batch_size=args.batch_size,
            num_epochs=args.num_epochs,
            shuffle=True
        )

        logging_hook = tf.train.LoggingTensorHook(
            tensors={
                "probabilities": "softmax"
            },
            every_n_iter=100
        )

        mnist_classifier.train(
            input_fn=train_input_fn,
            hooks=[logging_hook]
        )

    if args.eval:

        eval_input_fn = tf.estimator.inputs.numpy_input_fn(
            x={"images": eval_images},
            y=eval_labels,
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
            x={"images": eval_images},
            y=eval_labels,
            batch_size=args.batch_size,
            num_epochs=1,
            shuffle=False
        )

        predict_results = mnist_classifier.predict(
            input_fn=predict_input_fn
        )

        for predict_result in predict_results:

            image = predict_result["images"]
            attention = predict_result["attentions"]

            image = image.repeat(repeats=3, axis=-1)

            attention = np.apply_along_axis(func1d=np.sum, axis=-1, arr=attention)
            attention = utils.scale(attention, attention.min(), attention.max(), 0., 1.)

            image[:, :, -1] += attention

            cv2.imshow("image", cv2.resize(image, (128, 128)))

            if cv2.waitKey(1000) == ord("q"):

                break


if __name__ == "__main__":
    tf.app.run()
