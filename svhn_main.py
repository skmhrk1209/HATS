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
parser.add_argument("--steps", type=int, default=10000, help="training steps")
parser.add_argument("--epochs", type=int, default=100, help="training epochs")
parser.add_argument("--batch", type=int, default=100, help="batch size")
parser.add_argument("--model", type=str, default="svhn_acnn_model", help="model directory")
parser.add_argument('--train', action="store_true", help="with training")
parser.add_argument('--eval', action="store_true", help="with evaluation")
parser.add_argument('--predict', action="store_true", help="with prediction")
parser.add_argument('--gpu', type=str, default="0", help="gpu id")
args = parser.parse_args()

tf.logging.set_verbosity(tf.logging.INFO)


def svhn_input_fn(filenames, training, num_epochs=1, batch_size=1):

    def preprocess(image, training):

        image = tf.reshape(image, [64, 64, 3])
        image = tf.random_crop(image, [56, 56, 3])
        image = tf.image.convert_image_dtype(image, tf.float32)

        return image

    def parse(example, training):

        features = tf.parse_single_example(
            serialized=example,
            features={
                "image": tf.FixedLenFeature(
                    shape=[],
                    dtype=tf.string,
                    default_value=""
                ),
                "length": tf.FixedLenFeature(
                    shape=[1],
                    dtype=tf.int64,
                    default_value=[0]
                ),
                "digits": tf.FixedLenFeature(
                    shape=[5],
                    dtype=tf.int64,
                    default_value=[10] * 5
                )
            }
        )

        image = preprocess(tf.decode_raw(features["image"], tf.uint8), training)
        length = tf.cast(features['length'], tf.int32)
        digits = tf.cast(features['digits'], tf.int32)

        return {"images": image}, tf.concat([length, digits], 0)

    dataset = tf.data.TFRecordDataset(filenames)
    dataset = dataset.shuffle(30000)
    dataset = dataset.repeat(num_epochs)
    dataset = dataset.map(functools.partial(parse, training=training))
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(1)

    return dataset.make_one_shot_iterator().get_next()


def svhn_model_fn(features, labels, mode, params, size, data_format):
    """""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    model function for ACNN

    features:   batch of features from input_fn
    labels:     batch of labels from input_fn
    mode:       enum { TRAIN, EVAL, REDICT }
    """""""""""""""""""""""""""""""""""""""""""""""""""""""""""

    """""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    convolutional layer 1
    (-1, 56, 56, 1) -> (-1, 56, 56, 32)
    """""""""""""""""""""""""""""""""""""""""""""""""""""""""""

    inputs = features["images"]

    if data_format == "channels_first":

        inputs = tf.transpose(inputs, [0, 3, 1, 2])

    inputs = tf.layers.conv2d(
        inputs=inputs,
        filters=32,
        kernel_size=5,
        strides=1,
        padding="same",
        data_format=data_format
    )

    inputs = utils.batch_normalization(data_format)(
        inputs=inputs,
        training=mode == tf.estimator.ModeKeys.TRAIN,
        fused=True
    )

    inputs = tf.nn.relu(inputs)

    """""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    convolutional layer 2
    (-1, 56, 56, 32) -> (-1, 56, 56, 64)
    """""""""""""""""""""""""""""""""""""""""""""""""""""""""""

    inputs = tf.layers.conv2d(
        inputs=inputs,
        filters=64,
        kernel_size=5,
        strides=1,
        padding="same",
        data_format=data_format
    )

    inputs = utils.batch_normalization(data_format)(
        inputs=inputs,
        training=mode == tf.estimator.ModeKeys.TRAIN,
        fused=True
    )

    inputs = tf.nn.relu(inputs)

    """""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    attention convolutional layer 1
    (-1, 56, 56, 64) -> (-1, 28, 28, 3)
    """""""""""""""""""""""""""""""""""""""""""""""""""""""""""

    attentions = tf.layers.conv2d(
        inputs=inputs,
        filters=3,
        kernel_size=9,
        strides=2,
        padding="same",
        data_format=data_format
    )

    attentions = utils.batch_normalization(data_format)(
        inputs=attentions,
        training=mode == tf.estimator.ModeKeys.TRAIN,
        fused=True
    )

    attentions = tf.nn.relu(attentions)

    """""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    attention convolutional layer 2
    (-1, 28, 28, 3) -> (-1, 14, 14, 3)
    """""""""""""""""""""""""""""""""""""""""""""""""""""""""""

    attentions = tf.layers.conv2d(
        inputs=attentions,
        filters=3,
        kernel_size=9,
        strides=2,
        padding="same",
        data_format=data_format
    )

    attentions = utils.batch_normalization(data_format)(
        inputs=attentions,
        training=mode == tf.estimator.ModeKeys.TRAIN,
        fused=True
    )

    attentions = tf.nn.relu(attentions)

    """""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    attention dense layer 3
    (-1, 14, 14, 3) -> (-1, 10)
    """""""""""""""""""""""""""""""""""""""""""""""""""""""""""

    shape = attentions.get_shape().as_list()

    attentions = tf.layers.flatten(attentions)

    attentions = tf.layers.dense(
        inputs=attentions,
        units=10
    )

    attentions = tf.layers.batch_normalization(
        inputs=attentions,
        training=mode == tf.estimator.ModeKeys.TRAIN,
        fused=True
    )

    attentions = tf.nn.relu(attentions)

    """""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    attention dense layer 4
    (-1, 10) -> (-1, 14, 14, 3)
    """""""""""""""""""""""""""""""""""""""""""""""""""""""""""

    attentions = tf.layers.dense(
        inputs=attentions,
        units=functools.reduce(operator.mul, shape[1:])
    )

    attentions = tf.layers.batch_normalization(
        inputs=attentions,
        training=mode == tf.estimator.ModeKeys.TRAIN,
        fused=True
    )

    attentions = tf.nn.relu(attentions)

    attentions = tf.reshape(
        tensor=attentions,
        shape=[-1] + shape[1:]
    )

    """""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    attention deconvolutional layer 5
    (-1, 14, 14, 3) -> (-1, 28, 28, 9)
    """""""""""""""""""""""""""""""""""""""""""""""""""""""""""

    attentions = tf.layers.conv2d_transpose(
        inputs=attentions,
        filters=9,
        kernel_size=3,
        strides=2,
        padding="same",
        data_format=data_format
    )

    attentions = utils.batch_normalization(data_format)(
        inputs=attentions,
        training=mode == tf.estimator.ModeKeys.TRAIN,
        fused=True
    )

    attentions = tf.nn.relu(attentions)

    """""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    attention deconvolutional layer 6
    (-1, 28, 28, 9) -> (-1, 56, 56, 9)
    """""""""""""""""""""""""""""""""""""""""""""""""""""""""""

    attentions = tf.layers.conv2d_transpose(
        inputs=attentions,
        filters=9,
        kernel_size=3,
        strides=2,
        padding="same",
        data_format=data_format
    )

    attentions = utils.batch_normalization(data_format)(
        inputs=attentions,
        training=mode == tf.estimator.ModeKeys.TRAIN,
        fused=True
    )

    attentions = tf.nn.sigmoid(attentions)

    """""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    extract layer
    (-1, 56, 56, 64), (-1, 56, 56, 9) -> (-1, 64, 9)
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
    dense layer 3
    (-1, 64, 9) -> (-1, 1024)
    """""""""""""""""""""""""""""""""""""""""""""""""""""""""""

    inputs = tf.layers.flatten(inputs)

    inputs = tf.layers.dense(
        inputs=inputs,
        units=1024
    )

    inputs = tf.layers.batch_normalization(
        inputs=inputs,
        training=mode == tf.estimator.ModeKeys.TRAIN,
        fused=True
    )

    inputs = tf.nn.relu(inputs)

    """""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    logits layer 4
    (-1, 1024) -> (-1, 10)
    """""""""""""""""""""""""""""""""""""""""""""""""""""""""""

    length_logits = tf.layers.dense(
        inputs=inputs,
        units=6
    )

    digits_logits = [
        tf.layers.dense(
            inputs=inputs,
            units=11
        )
    ] * 5

    predictions = {
        "length_classes": tf.argmax(
            input=length_logits,
            axis=1
        ),
        "length_probabilities": tf.nn.softmax(
            logits=length_logits,
            dim=1,
            name="length_softmax"
        ),
        "digits_classes": tf.stack(
            values=[
                tf.argmax(
                    input=digit_logits,
                    axis=1
                ) for digit_logits in digits_logits
            ],
            axis=1
        ),
        "digits_probabilities": tf.stack(
            values=[
                tf.nn.softmax(
                    logits=digit_logits,
                    dim=1
                ) for digit_logits in digits_logits
            ],
            axis=1,
            name="digits_softmax"
        ),
        "attentions": (lambda cond, func, inputs: func(inputs) if cond else inputs)(
            cond=data_format == "channels_first",
            func=functools.partial(tf.transpose, perm=[0, 2, 3, 1]),
            inputs=utils.chunk_images(
                inputs=attentions,
                size=size,
                data_format=data_format
            )
        )
    }

    predictions.update(features)

    if mode == tf.estimator.ModeKeys.PREDICT:

        return tf.estimator.EstimatorSpec(
            mode=mode,
            predictions=predictions
        )

    length_loss = tf.losses.sparse_softmax_cross_entropy(
        labels=labels[:, 0],
        logits=length_logits
    )

    digits_loss = tf.reduce_sum(
        input_tensor=[
            tf.losses.sparse_softmax_cross_entropy(
                labels=labels[:, i],
                logits=digit_logits
            ) for i, digit_logits in enumerate(digits_logits)
        ],
        axis=None
    )

    loss = length_loss + digits_loss

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

    svhn_classifier = tf.estimator.Estimator(
        model_fn=functools.partial(
            svhn_model_fn,
            size=[56, 56],
            data_format="channels_first"
        ),
        model_dir=args.model,
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

        train_input_fn = functools.partial(
            svhn_input_fn,
            filenames=["data/train.tfrecords"],
            training=True,
            num_epochs=args.epochs,
            batch_size=args.batch
        )

        logging_hook = tf.train.LoggingTensorHook(
            tensors={
                "length_probabilities": "length_softmax",
                "digits_probabilities": "digits_softmax"
            },
            every_n_iter=100
        )

        svhn_classifier.train(
            input_fn=train_input_fn,
            hooks=[logging_hook],
            steps=args.steps
        )

    if args.eval:

        eval_input_fn = functools.partial(
            svhn_input_fn,
            filenames=["data/test.tfrecords"],
            training=False,
            num_epochs=1,
            batch_size=args.batch
        )

        eval_results = svhn_classifier.evaluate(
            input_fn=eval_input_fn
        )

        print(eval_results)

    if args.predict:

        predict_input_fn = functools.partial(
            svhn_input_fn,
            filenames=["test.tfrecords"],
            training=False,
            num_epochs=1,
            batch_size=args.batch
        )

        predict_results = svhn_classifier.predict(
            input_fn=predict_input_fn
        )

        for predict_result in predict_results:

            image = predict_result["images"]
            attention = predict_result["attentions"]

            image = image.repeat(repeats=3, axis=-1)

            attention = utils.scale(attention, attention.min(), attention.max(), 0., 1.)
            attention = np.apply_along_axis(func1d=np.sum, axis=-1, arr=attention)

            image[:, :, -1] += attention

            cv2.imshow("image", cv2.resize(image, (112, 112)))

            if cv2.waitKey(1000) == ord("q"):

                break


if __name__ == "__main__":
    tf.app.run()
