# =============================================================
# dataset details
# dataset: synth90k
# download: http://www.robots.ox.ac.uk/~vgg/data/text/
# train: 7224612
# val: 802734
# test: 891927
# max num chars: 23
# classes: [0-9A-Z!(eos)](case-insensitive)
# word accuracy:
# edit distance:
# pretrained model: chars74k classifier
# max steps: 100000 batch size: 128
# =============================================================

import tensorflow as tf
import optuna
import argparse
import functools
import dataset
from attrdict import AttrDict
from models.hats import HATS
from networks.attention_network import AttentionNetwork
from networks.resnet import ResNet
from algorithms import *

parser = argparse.ArgumentParser()
parser.add_argument("--model_dir", type=str, default="mnist_hats_model", help="model directory")
parser.add_argument("--pretrained_model_dir", type=str, default="", help="pretrained model directory")
parser.add_argument("--batch_size", type=int, default=100, help="batch size")
parser.add_argument("--random_seed", type=int, default=1209, help="random seed")
parser.add_argument("--data_format", type=str, default="channels_first", help="data format")
parser.add_argument("--max_steps", type=int, default=50000, help="maximum number of training steps")
parser.add_argument("--steps", type=int, default=None, help="number of evaluation steps")
parser.add_argument('--train', action="store_true", help="with training")
parser.add_argument('--eval', action="store_true", help="with evaluation")
parser.add_argument("--gpu", type=str, default="0", help="gpu id")
args = parser.parse_args()

tf.logging.set_verbosity(tf.logging.INFO)


if __name__ == "__main__":

    estimator = tf.estimator.Estimator(
        model_fn=lambda features, labels, mode: HATS(
            backbone_network=PyramidResNet(
                conv_param=AttrDict(filters=64, kernel_size=[7, 7], strides=[2, 2]),
                pool_param=None,
                residual_params=[
                    AttrDict(filters=32, strides=[2, 2], blocks=1),
                    AttrDict(filters=64, strides=[2, 2], blocks=1),
                ],
                data_format=args.data_format
            ),
            attention_network=AttentionNetwork(
                conv_params=[
                    AttrDict(filters=8, kernel_size=[3, 3], strides=[2, 2]),
                    AttrDict(filters=8, kernel_size=[3, 3], strides=[2, 2]),
                ],
                rnn_params=[
                    AttrDict(sequence_length=5, num_units=256),
                ],
                deconv_params=[
                    AttrDict(filters=8, kernel_size=[3, 3], strides=[2, 2]),
                    AttrDict(filters=8, kernel_size=[3, 3], strides=[2, 2]),
                ],
                data_format=args.data_format
            ),
            num_units=[128],
            num_classes=11,
            data_format=args.data_format,
            hyper_params=AttrDict(
                optimizer=tf.train.AdamOptimizer()
            )
        )(features, labels, mode),
        model_dir=args.model_dir,
        config=tf.estimator.RunConfig(
            tf_random_seed=args.random_seed,
            save_summary_steps=100,
            save_checkpoints_steps=100,
            session_config=tf.ConfigProto(
                gpu_options=tf.GPUOptions(
                    visible_device_list=args.gpu,
                    allow_growth=True
                )
            )
        )
    )

    if args.train:

        estimator.train(
            input_fn=functools.partial(
                dataset.input_fn,
                filenames=args.train_filenames,
                batch_size=args.batch_size,
                num_epochs=None,
                shuffle=True,
                sequence_lengths=[5],
                encoding="jpeg",
                image_size=[256, 256],
                data_format=args.data_format
            ),
            max_steps=args.max_steps,
            hooks=[
                tf.train.LoggingTensorHook(
                    tensors={
                        "word_accuracy": "word_accuracy"
                    },
                    every_n_iter=100
                )
            ]
        )

    if args.eval:

        print(estimator.evaluate(
            input_fn=functools.partial(
                dataset.input_fn,
                filenames=args.test_filenames,
                batch_size=args.batch_size,
                num_epochs=1,
                shuffle=False,
                sequence_lengths=[5],
                encoding="jpeg",
                image_size=[256, 256],
                data_format=args.data_format
            ),
            steps=args.steps
        ))
