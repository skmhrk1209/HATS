from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as ani
import tensorflow as tf
import argparse
import cv2

parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str,
                    default="mnist_acnn_model", help="model directory")
parser.add_argument("--batch", type=int, default=100, help="batch size")
parser.add_argument("--steps", type=int, default=10000, help="training steps")
parser.add_argument('--train', action='store_true', help='with training')
parser.add_argument('--eval', action='store_true', help='with evaluation')
parser.add_argument('--predict', action='store_true', help='with prediction')
args = parser.parse_args()

tf.logging.set_verbosity(tf.logging.INFO)


def acnn_model_fn(features, labels, mode):
    """""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    model function for ACNN

    features:   batch of features from input_fn
    labels:     batch of labels from input_fn
    mode:       enum { TRAIN, EVAL, PREDICT }
    """""""""""""""""""""""""""""""""""""""""""""""""""""""""""

    """""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    convolutional layer 1
    (-1, 64, 64, 1) -> (-1, 64, 64, 32)
    """""""""""""""""""""""""""""""""""""""""""""""""""""""""""

    conv1 = tf.layers.conv2d(
        inputs=features["x"],
        filters=32,
        kernel_size=(3, 3),
        strides=(1, 1),
        padding="same",
        activation=tf.nn.relu
    )

    """""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    convolutional layer 2
    (-1, 64, 64, 32) -> (-1, 32, 32, 64)
    """""""""""""""""""""""""""""""""""""""""""""""""""""""""""

    conv2 = tf.layers.conv2d(
        inputs=conv1,
        filters=64,
        kernel_size=(3, 3),
        strides=(2, 2),
        padding="same",
        activation=tf.nn.relu
    )

    """""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    attention convolutional layer 3
    (-1, 32, 32, 64) -> (-1, 16, 16, 3)
    """""""""""""""""""""""""""""""""""""""""""""""""""""""""""

    conv3 = tf.layers.conv2d(
        inputs=conv2,
        filters=3,
        kernel_size=(9, 9),
        strides=(2, 2),
        padding="same",
        activation=tf.nn.relu
    )

    """""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    attention convolutional layer 4
    (-1, 16, 16, 3) -> (-1, 8, 8, 3)
    """""""""""""""""""""""""""""""""""""""""""""""""""""""""""

    conv4 = tf.layers.conv2d(
        inputs=conv3,
        filters=3,
        kernel_size=(9, 9),
        strides=(2, 2),
        padding="same",
        activation=tf.nn.relu
    )

    """""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    attention dense layer 5
    (-1, 8, 8, 3) -> (-1, 10)
    """""""""""""""""""""""""""""""""""""""""""""""""""""""""""

    dense5 = tf.layers.dense(
        inputs=tf.reshape(
            tensor=conv4,
            shape=(-1, 8 * 8 * 3)
        ),
        units=10,
        activation=tf.nn.relu
    )

    """""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    attention dense layer 6
    (-1, 10) -> (-1, 8, 8, 3)
    """""""""""""""""""""""""""""""""""""""""""""""""""""""""""

    dense6 = tf.layers.dense(
        inputs=dense5,
        units=8 * 8 * 3,
        activation=tf.nn.relu
    )

    """""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    attention deconvolutional layer 7
    (-1, 8, 8, 3) -> (-1, 16, 16, 9)
    """""""""""""""""""""""""""""""""""""""""""""""""""""""""""

    deconv7 = tf.layers.conv2d_transpose(
        inputs=tf.reshape(
            tensor=dense6,
            shape=(-1, 8, 8, 3)
        ),
        filters=9,
        kernel_size=(3, 3),
        strides=(2, 2),
        padding="same",
        activation=tf.nn.relu
    )

    """""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    attention deconvolutional layer 8
    (-1, 16, 16, 9) -> (-1, 32, 32, 9)
    """""""""""""""""""""""""""""""""""""""""""""""""""""""""""

    attentions = tf.layers.conv2d_transpose(
        inputs=deconv7,
        filters=9,
        kernel_size=(3, 3),
        strides=(2, 2),
        padding="same",
        activation=tf.nn.sigmoid
    )

    """""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    extract layer 9
    (-1, 32, 32, 64), (-1, 32, 32, 9) -> (-1, 64, 9)
    """""""""""""""""""""""""""""""""""""""""""""""""""""""""""

    matmul9 = tf.matmul(
        a=tf.reshape(
            tensor=conv2,
            shape=(-1, 32 * 32, 64)
        ),
        b=tf.reshape(
            tensor=attentions,
            shape=(-1, 32 * 32, 9)
        ),
        transpose_a=True,
        transpose_b=False
    )

    """""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    dense layer 10
    (-1, 64, 9) -> (-1, 128)
    """""""""""""""""""""""""""""""""""""""""""""""""""""""""""

    dense10 = tf.layers.dense(
        inputs=tf.reshape(
            tensor=matmul9,
            shape=(-1, 64 * 9)
        ),
        units=128,
        activation=tf.nn.relu
    )

    """""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    logits layer 4
    (-1, 128) -> (-1, 10)
    """""""""""""""""""""""""""""""""""""""""""""""""""""""""""

    logits = tf.layers.dense(
        inputs=dense10,
        units=10
    )

    predictions = {
        "classes": tf.argmax(
            input=logits,
            axis=1
        ),
        "probabilities": tf.nn.softmax(
            logits=logits,
            name="softmax_tensor"
        ),
        "images": features["x"],
        "attentions": attentions
    }

    if mode == tf.estimator.ModeKeys.PREDICT:

        return tf.estimator.EstimatorSpec(
            mode=mode,
            predictions=predictions
        )

    loss = tf.losses.sparse_softmax_cross_entropy(
        labels=labels,
        logits=logits
    )

    """""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    IMPORTANT !!!
    """""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    loss += tf.reduce_sum(tf.abs(attentions)) * 1e-6

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

    def scale(inVal, inMin, inMax, outMin, outMax): return outMin + \
        (inVal - inMin) / (inMax - inMin) * (outMax - outMin)

    mnist = tf.contrib.learn.datasets.load_dataset("mnist")
    train_data = np.zeros(
        (mnist.train.images.shape[0], 64, 64, 1), dtype=np.float32)
    eval_data = np.zeros(
        (mnist.test.images.shape[0], 64, 64, 1), dtype=np.float32)
    train_labels = mnist.train.labels.astype(np.int32)
    eval_labels = mnist.test.labels.astype(np.int32)

    for translated, raw in zip(train_data, mnist.train.images):

        x = np.random.randint(36)
        y = np.random.randint(36)

        translated[y:y+28, x:x+28] = raw.reshape((28, 28, 1))

    for translated, raw in zip(eval_data, mnist.test.images):

        x = np.random.randint(36)
        y = np.random.randint(36)

        translated[y:y+28, x:x+28] = raw.reshape((28, 28, 1))

    run_config = tf.estimator.RunConfig().replace(
        session_config=tf.ConfigProto(device_count={'GPU': 1}))

    mnist_classifier = tf.estimator.Estimator(
        model_fn=acnn_model_fn,
        model_dir=args.model,
        config=run_config
    )

    if args.train:

        train_input_fn = tf.estimator.inputs.numpy_input_fn(
            x={"x": train_data},
            y=train_labels,
            batch_size=args.batch,
            num_epochs=None,
            shuffle=True
        )

        logging_hook = tf.train.LoggingTensorHook(
            tensors={
                "probabilities": "softmax_tensor"
            },
            every_n_iter=100
        )

        mnist_classifier.train(
            input_fn=train_input_fn,
            steps=args.steps,
            hooks=[logging_hook]
        )

    if args.eval:

        eval_input_fn = tf.estimator.inputs.numpy_input_fn(
            x={"x": eval_data},
            y=eval_labels,
            num_epochs=1,
            shuffle=False
        )

        eval_results = mnist_classifier.evaluate(
            input_fn=eval_input_fn
        )

        print(eval_results)

    if args.predict:

        predict_input_fn = tf.estimator.inputs.numpy_input_fn(
            x={"x": eval_data[:10]},
            num_epochs=1,
            shuffle=False
        )

        predict_results = mnist_classifier.predict(
            input_fn=predict_input_fn
        )

        figure = plt.figure()
        images = []

        for predict_result in predict_results:

            image = predict_result["images"].repeat(3, axis=-1)

            attention = np.apply_along_axis(
                np.sum, axis=-1, arr=predict_result["attentions"])

            attention = scale(attention, attention.min(),
                              attention.max(), 0, 1)

            attention = cv2.resize(attention, (64, 64))

            image[:, :, 0] += attention

            images.append([plt.imshow(image, animated=True)])

        animation = ani.ArtistAnimation(
            figure, images, interval=1000, repeat=False)

        plt.show()


if __name__ == "__main__":
    tf.app.run()
