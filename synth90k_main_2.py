# =============================================================
# dataset details
# dataset: synth90k
# download: http://www.robots.ox.ac.uk/~vgg/data/text/
# train: 7224612
# val: 802734
# test: 891927
# max num chars: 23
# num classes: 37 (only alphanumeric characters, case-insensitive)
# =============================================================


import tensorflow as tf
import tensorflow_hub as hub
import argparse
from attrdict import AttrDict
from dataset import Dataset
from model import HATS
from networks.han import HAN
from networks.resnet import ResNet
from algorithms import *

parser = argparse.ArgumentParser()
parser.add_argument("--model_dir", type=str, default="synth90k_hats_model_2", help="model directory")
parser.add_argument("--pretrained_model_dir", type=str, default="", help="pretrained model directory")
parser.add_argument('--filenames', type=str, nargs="+", default=["synth90k_train.tfrecord"], help="tfrecord filenames")
parser.add_argument("--num_epochs", type=int, default=1, help="number of training epochs")
parser.add_argument("--batch_size", type=int, default=128, help="batch size")
parser.add_argument("--data_format", type=str, default="channels_first", help="data format")
parser.add_argument("--steps", type=int, default=None, help="number of training epochs")
parser.add_argument("--max_steps", type=int, default=None, help="maximum number of training epochs")
parser.add_argument("--train", action="store_true", help="with training")
parser.add_argument("--eval", action="store_true", help="with evaluation")
parser.add_argument("--predict", action="store_true", help="with prediction")
parser.add_argument("--gpu", type=str, default="0,1,2", help="gpu id")
parser.add_argument("--random_seed", type=int, default=1209, help="random seed")
args = parser.parse_args()

tf.logging.set_verbosity(tf.logging.INFO)


def pyramid_resnet(inputs, training):

    resnet = hub.Module(
        spec="https://tfhub.dev/google/imagenet/resnet_v2_50/feature_vector/1",
        trainable=True,
        name="resnet",
        tags={"train"} if training else None
    )

    feature_maps = [resnet(
        inputs=dict(images=inputs),
        signature="image_feature_vector",
        as_dict=True
    )[name] for name in [f"resnet_v2_50/block{i}" for i in range(1, 4)]]

    inputs = feature_maps.pop()

    while feature_maps:

        inputs = tf.image.resize_bilinear(inputs, tf.shape(inputs)[1:-1] * 2)

        inputs = tf.layers.conv2d(
            inputs=inputs,
            filters=feature_maps[-1].get_shape().as_list()[1],
            kernel_size=[1, 1],
            strides=[1, 1],
            padding="same",
            data_format="channels_last",
            use_bias=False,
            kernel_initializer=tf.variance_scaling_initializer(
                scale=2.0,
                mode="fan_in",
                distribution="normal"
            )
        )

        inputs += feature_maps.pop()

    return inputs


def main(unused_argv):

    classifier = tf.estimator.Estimator(
        model_fn=lambda features, labels, mode: HATS(
            backbone_network=pyramid_resnet,
            attention_network=HAN(
                conv_params=[
                    AttrDict(filters=16, kernel_size=[3, 3], strides=[2, 2]),
                    AttrDict(filters=16, kernel_size=[3, 3], strides=[2, 2]),
                ],
                deconv_params=[
                    AttrDict(filters=16, kernel_size=[3, 3], strides=[2, 2]),
                    AttrDict(filters=16, kernel_size=[3, 3], strides=[2, 2]),
                ],
                rnn_params=[
                    AttrDict(sequence_length=23, num_units=256)
                ],
                data_format=args.data_format
            ),
            num_classes=37,
            data_format=args.data_format,
            hyper_params=AttrDict(
                attention_decay=1e-4,
                learning_rate=0.001,
                beta1=0.9,
                beta2=0.999
            )
        )(features, labels, mode),
        model_dir=args.model_dir,
        config=tf.estimator.RunConfig(
            tf_random_seed=args.random_seed,
            session_config=tf.ConfigProto(
                gpu_options=tf.GPUOptions(
                    visible_device_list=args.gpu,
                    allow_growth=True
                )
            )
        )
    )

    if args.train:

        classifier.train(
            input_fn=Dataset(
                filenames=args.filenames,
                num_epochs=args.num_epochs,
                batch_size=args.batch_size,
                sequence_lengths=[23],
                image_size=[256, 256],
                data_format=args.data_format
            ),
            steps=args.steps,
            max_steps=args.max_steps,
            hooks=[
                tf.train.LoggingTensorHook(
                    tensors={"accuracy": "accuracy"},
                    every_n_iter=100
                )
            ]
        )

    if args.eval:

        eval_results = classifier.evaluate(
            input_fn=Dataset(
                filenames=args.filenames,
                num_epochs=1,
                batch_size=args.batch_size,
                sequence_lengths=[23],
                image_size=[256, 256],
                data_format=args.data_format
            )
        )

        print(eval_results)


if __name__ == "__main__":
    tf.app.run()
