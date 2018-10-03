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
import os
import glob
from attention_network import AttentionNetwork
from attr_dict import AttrDict

parser = argparse.ArgumentParser()
parser.add_argument("--model_dir", type=str, default="mnist_acnn_model", help="model directory")
parser.add_argument("--num_epochs", type=int, default=100, help="number of training epochs")
parser.add_argument("--batch_size", type=int, default=100, help="batch size")
parser.add_argument('--train', action="store_true", help="with training")
parser.add_argument('--eval', action="store_true", help="with evaluation")
parser.add_argument('--predict', action="store_true", help="with prediction")
parser.add_argument('--gpu', type=str, default="0", help="gpu id")
args = parser.parse_args()

tf.logging.set_verbosity(tf.logging.INFO)


def acnn_model_fn(features, labels, mode, params):
    """""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    model function for ACNN
    features:   batch of features from input_fn
    labels:     batch of labels from input_fn
    mode:       enum { TRAIN, EVAL, PREDICT }
    """""""""""""""""""""""""""""""""""""""""""""""""""""""""""

    params = AttrDict(params)
    predictions = features.copy()

    inputs = features["images"]

    with tf.variable_scope("acnn"):

        inputs = tf.layers.conv2d(
            inputs=inputs,
            filters=32,
            kernel_size=3,
            strides=2,
            padding="same",
            activation=tf.nn.relu
        )

        inputs = tf.layers.conv2d(
            inputs=inputs,
            filters=64,
            kernel_size=3,
            strides=2,
            padding="same",
            activation=tf.nn.relu
        )

        attention_network = AttentionNetwork(
            conv_params=[
                AttrDict(
                    filters=3,
                    kernel_size=9,
                    strides=2
                ),
                AttrDict(
                    filters=3,
                    kernel_size=9,
                    strides=2
                )
            ],
            deconv_params=[
                AttrDict(
                    filters=9,
                    kernel_size=3,
                    strides=2
                ),
                AttrDict(
                    filters=9,
                    kernel_size=3,
                    strides=2
                )
            ],
            bottleneck_units=10,
            data_format="channels_last"
        )

        attentions = attention_network(
            inputs=inputs,
            training=mode == tf.estimator.ModeKeys.TRAIN
        )

        attentions = tf.cond(
            pred=params.training_attention,
            true_fn=lambda: attentions,
            false_fn=lambda: tf.ones_like(attentions)
        )

        predictions["attentions"] = attentions

        shape = inputs.shape.as_list()

        inputs = tf.reshape(
            tensor=inputs,
            shape=[-1, np.prod(shape[1:3]), shape[3]]
        )

        shape = attentions.shape.as_list()

        attentions = tf.reshape(
            tensor=attentions,
            shape=[-1, np.prod(shape[1:3]), shape[3]]
        )

        inputs = tf.matmul(
            a=inputs,
            b=attentions,
            transpose_a=True,
            transpose_b=False
        )

        inputs = tf.layers.flatten(inputs)

        inputs = tf.layers.dense(
            inputs=inputs,
            units=128,
            activation=tf.nn.relu
        )

        logits = tf.layers.dense(
            inputs=inputs,
            units=10
        )

    predictions.update({
        "classes": tf.argmax(
            input=logits,
            axis=-1
        ),
        "softmax": tf.nn.softmax(
            logits=logits,
            dim=-1,
            name="softmax"
        )
    })

    tf.summary.image("images", predictions["images"], 10)
    tf.summary.image("attentions", predictions["attentions"], 10)

    print("num params: {}".format(
        np.sum([np.prod(variable.get_shape().as_list())
                for variable in tf.global_variables("acnn")])
    ))

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
        input_tensor=tf.abs(attentions),
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

    def load_mnist(path):

        filenames = glob.glob(os.path.join(path, "*.png"))

        images = np.array([cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
                           for filename in filenames], dtype=np.float32)
        images = np.reshape(images, [-1, 128, 128, 1])
        images /= 255.0

        labels = np.array([int(os.path.splitext(os.path.basename(filename))[0].split("-")[-1])
                           for filename in filenames], dtype=np.int32)

        return images, labels

    train_images, train_labels = load_mnist("data/mnist/train")
    test_images, test_labels = load_mnist("data/mnist/test")

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
            "attention_decay": 1e-6,
            "training_attention": tf.constant(False)
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
                "softmax": "softmax"
            },
            every_n_iter=100
        )

        mnist_classifier.train(
            input_fn=train_input_fn,
            hooks=[logging_hook]
        )

    if args.eval:

        eval_input_fn = tf.estimator.inputs.numpy_input_fn(
            x={"images": test_images},
            y=test_labels,
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
            x={"images": test_images},
            y=test_labels,
            batch_size=args.batch_size,
            num_epochs=1,
            shuffle=False
        )

        predict_results = mnist_classifier.predict(
            input_fn=predict_input_fn
        )

        for i, predict_result in enumerate(itertools.islice(predict_results, 10)):

            image = predict_result["images"]
            attentions = predict_result["attentions"]

            attention = np.apply_along_axis(np.sum, axis=3, arr=attentions)
            attention = cv2.resize(attention, (128, 128))

            image = image.repeat(3, axis=3)
            image[:, :, 2] += attention

            cv2.imshow("image", image)

            if cv2.waitKey(1000) == ord("q"):
                break


if __name__ == "__main__":
    tf.app.run()
