from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import cv2
import os
import argparse
from attention_network import AttentionNetwork

parser = argparse.ArgumentParser()
parser.add_argument("--model_dir", type=str, default="imagenet_acnn_model", help="model directory")
parser.add_argument("--filenames", type=str, nargs="+", default=["train.tfrecord"], help="tfrecord filenames")
parser.add_argument("--num_epochs", type=int, default=100, help="number of training epochs")
parser.add_argument("--batch_size", type=int, default=100, help="batch size")
parser.add_argument('--train', action="store_true", help="with training")
parser.add_argument('--eval', action="store_true", help="with evaluation")
parser.add_argument('--predict', action="store_true", help="with prediction")
parser.add_argument('--gpu', type=str, default="0", help="gpu id")
args = parser.parse_args()

tf.logging.set_verbosity(tf.logging.INFO)


class AttrDict(dict):

    def __getattr__(self, name): return self[name]

    def __setattr__(self, name, value): self[name] = value

    def __delattr__(self, name): del self[name]


urls = AttrDict(
    resnet_v1_50="https://tfhub.dev/google/imagenet/resnet_v1_50/feature_vector/1",
    resnet_v1_101="https://tfhub.dev/google/imagenet/resnet_v1_101/feature_vector/1",
    resnet_v1_152="https://tfhub.dev/google/imagenet/resnet_v1_152/feature_vector/1",
    resnet_v2_50="https://tfhub.dev/google/imagenet/resnet_v2_50/feature_vector/1",
    resnet_v2_101="https://tfhub.dev/google/imagenet/resnet_v2_101/feature_vector/1",
    resnet_v2_152="https://tfhub.dev/google/imagenet/resnet_v2_152/feature_vector/1"
)


def imagenet_input_fn(filenames, num_epochs, batch_size, buffer_size):

    def parse(example):

        features = tf.parse_single_example(
            serialized=example,
            features={
                "path": tf.FixedLenFeature(
                    shape=[],
                    dtype=tf.string,
                    default_value=""
                ),
                "label": tf.FixedLenFeature(
                    shape=[],
                    dtype=tf.int64,
                    default_value=0
                )
            }
        )

        image = tf.read_file(features["path"])
        image = tf.image.decode_jpeg(image, 3)
        image = tf.image.convert_image_dtype(image, tf.float32)

        label = tf.cast(features["label"], tf.int32)

        return image, label

    dataset = tf.data.TFRecordDataset(filenames)
    dataset = dataset.shuffle(buffer_size)
    dataset = dataset.repeat(num_epochs)
    dataset = dataset.map(parse)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(1)

    return dataset.make_one_shot_iterator()


def acnn_model_fn(features, labels, mode, params):

    params = AttrDict(params)
    predictions = features.copy()

    images = features["images"]

    resnet_v2_50 = hub.Module(urls.resnet_v2_50)

    attention_network = AttentionNetwork(
        conv_params=[
            AttrDict(
                filters=4,
                kernel_size=9,
                strides=1
            )
        ] * 2,
        deconv_params=[
            AttrDict(
                filters=16,
                kernel_size=3,
                strides=1
            )
        ] * 2,
        bottleneck_units=128,
        data_format="channels_last"
    )

    feature_maps = resnet_v2_50(
        dict(images=images),
        signature="image_feature_vector",
        as_dict=True
    )["resnet_v2_50/block4"]

    attentions = attention_network(
        inputs=feature_maps,
        training=mode == tf.estimator.ModeKeys.TRAIN
    )

    predictions["attentions"] = tf.reduce_sum(
        input_tensor=attentions,
        axis=3,
        keep_dims=True
    )

    shape = feature_maps.shape.as_list()

    feature_maps = tf.reshape(
        tensor=feature_maps,
        shape=[-1, np.prod(shape[1:3]), shape[3]]
    )

    shape = attentions.shape.as_list()

    attentions = tf.reshape(
        tensor=attentions,
        shape=[-1, np.prod(shape[1:3]), shape[3]]
    )

    feature_vectors = tf.matmul(
        a=feature_maps,
        b=attentions,
        transpose_a=True,
        transpose_b=False
    )

    feature_vectors = tf.layers.flatten(feature_vectors)

    feature_vectors = tf.layers.dense(
        inputs=feature_vectors,
        units=1024,
        activation=tf.nn.relu
    )

    logits = tf.layers.dense(
        inputs=feature_vectors,
        units=1000
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

    imagenet_classifier = tf.estimator.Estimator(
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
        params=dict(
            attention_decay=1e-6,
            training_attention=True
        )
    )

    if args.train:

        logging_hook = tf.train.LoggingTensorHook(
            tensors={
                "softmax": "softmax"
            },
            every_n_iter=100
        )

        imagenet_classifier.train(
            input_fn=lambda: imagenet_input_fn(
                filenames=args.filenames,
                num_epochs=args.num_epochs,
                batch_size=args.batch_size,
                buffer_size=args.buffer_size
            ),
            hooks=[logging_hook]
        )

    if args.eval:

        eval_results = imagenet_classifier.evaluate(
            input_fn=lambda: imagenet_input_fn(
                filenames=args.filenames,
                num_epochs=args.num_epochs,
                batch_size=args.batch_size,
                buffer_size=args.buffer_size
            )
        )

        print(eval_results)


if __name__ == "__main__":
    tf.app.run()
