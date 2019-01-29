''' synth90k dataset details
    url: http://www.robots.ox.ac.uk/~vgg/data/text/
    train: 7224612
    val: 802734
    test: 891927
    max num chars: 23
    num classes: 37 (case-insensitive)

    broken files included:
    synth90k/2911/6/77_heretical_35885.jpg
    synth90k/2852/6/60_TOILSOME_79481.jpg
    synth90k/2749/6/101_Chided_13155.jpg
    synth90k/2069/4/192_whittier_86389.jpg
    synth90k/2025/2/364_SNORTERS_72304.jpg
    synth90k/2013/2/370_refract_63890.jpg
    synth90k/1881/4/225_Marbling_46673.jpg
    synth90k/1863/4/223_Diligently_21672.jpg
    synth90k/1817/2/363_actuating_904.jpg
    synth90k/913/4/231_randoms_62372.jpg
    synth90k/869/4/234_TRIASSIC_80582.jpg
    synth90k/495/6/81_MIDYEAR_48332.jpg
    synth90k/368/4/232_friar_30876.jpg
    synth90k/275/6/96_hackle_34465.jpg
    synth90k/173/2/358_BURROWING_10395.jpg
'''

import tensorflow as tf
import numpy as np
import argparse
import functools
import itertools
import glob
import os
import cv2
from attrdict import AttrDict
from dataset import Dataset
from model import HATS
from networks.han import HAN
from networks.resnet import ResNet
from algorithms import *

parser = argparse.ArgumentParser()
parser.add_argument("--model_dir", type=str, default="synth90k_hats_model", help="model directory")
parser.add_argument("--pretrained_model_dir", type=str, default="", help="pretrained model directory")
parser.add_argument('--filenames', type=str, nargs="+", default=["synth90k_train.tfrecord"], help="tfrecord filenames")
parser.add_argument("--num_epochs", type=int, default=1, help="number of training epochs")
parser.add_argument("--batch_size", type=int, default=128, help="batch size")
parser.add_argument("--buffer_size", type=int, default=7224612, help="buffer size to shuffle dataset")
parser.add_argument("--data_format", type=str, default="channels_first", help="data format")
parser.add_argument("--train", action="store_true", help="with training")
parser.add_argument("--eval", action="store_true", help="with evaluation")
parser.add_argument("--predict", action="store_true", help="with prediction")
parser.add_argument("--gpu", type=str, default="0,1,2", help="gpu id")
parser.add_argument("--random_seed", type=int, default=1209, help="random seed")
args = parser.parse_args()

tf.logging.set_verbosity(tf.logging.INFO)


def main(unused_argv):

    classifier = tf.estimator.Estimator(
        model_fn=lambda features, labels, mode: HATS(
            backbone_network=ResNet(
                conv_param=AttrDict(filters=64, kernel_size=[7, 7], strides=[2, 2]),
                pool_param=None,
                residual_params=[
                    AttrDict(filters=64, strides=[2, 2], blocks=2),
                    AttrDict(filters=128, strides=[2, 2], blocks=2),
                ],
                data_format=args.data_format
            ),
            attention_network=HAN(
                conv_params=[
                    AttrDict(filters=4, kernel_size=[9, 9], strides=[2, 2]),
                    AttrDict(filters=4, kernel_size=[9, 9], strides=[2, 2]),
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
                buffer_size=args.buffer_size,
                sequence_lengths=[23],
                image_size=[256, 256],
                data_format=args.data_format
            ),
            hooks=[
                tf.train.LoggingTensorHook(
                    tensors={"error_rate": "error_rate"},
                    every_n_iter=100
                )
            ]
        )

    if args.eval:

        eval_results = classifier.evaluate(
            input_fn=Dataset(
                filenames=args.filenames,
                num_epochs=args.num_epochs,
                batch_size=args.batch_size,
                buffer_size=args.buffer_size,
                sequence_lengths=[23],
                image_size=[256, 256],
                data_format=args.data_format
            )
        )

        print(eval_results)


if __name__ == "__main__":
    tf.app.run()
