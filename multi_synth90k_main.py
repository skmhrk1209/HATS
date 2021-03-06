# =============================================================
# dataset details
# dataset: Multi-Synth90k
# download: made by make_multi_synth90k.cpp
# train: 800000
# val: 100000
# test: 100000
# max num words: 5
# max num chars: 10
# classes: 37 [0-9A-Z](case-insensitive)
# word accuracy: 79.24 %
# edit distance: 6.603 %
# pretrained model: Chars74k classifier
# max steps: 100000 batch size: 50
# =============================================================

import tensorflow as tf
import numpy as np
import skimage
import argparse
import functools
import itertools
import dataset
import hooks
from models.hats import HATS
from networks.attention_network import AttentionNetwork
from networks.pyramid_resnet import PyramidResNet
from attrdict import AttrDict as Param
from algorithms import *

parser = argparse.ArgumentParser()
parser.add_argument("--model_dir", type=str, default="multi_synth90k_hats_model", help="model directory")
parser.add_argument("--pretrained_model_dir", type=str, default="chars74k_classifier", help="pretrained model directory")
parser.add_argument('--train_filenames', type=str, nargs="+", default=["multi_synth90k_train.tfrecord"], help="tfrecords for training")
parser.add_argument('--val_filenames', type=str, nargs="+", default=["multi_synth90k_val.tfrecord"], help="tfrecords for validation")
parser.add_argument('--test_filenames', type=str, nargs="+", default=["multi_synth90k_test.tfrecord"], help="tfrecords for test")
parser.add_argument("--batch_size", type=int, default=50, help="batch size")
parser.add_argument("--random_seed", type=int, default=1209, help="random seed")
parser.add_argument("--data_format", type=str, default="channels_first", help="data format")
parser.add_argument("--max_steps", type=int, default=100000, help="maximum number of training steps")
parser.add_argument("--steps", type=int, default=None, help="number of test steps")
parser.add_argument('--train', action="store_true", help="with training")
parser.add_argument('--eval', action="store_true", help="with evaluation")
parser.add_argument('--predict', action="store_true", help="with prediction")
parser.add_argument("--gpu", type=str, default="0", help="gpu id")
args = parser.parse_args()

tf.logging.set_verbosity(tf.logging.INFO)


if __name__ == "__main__":

    # validation時のbatch normalizationの統計は
    # ミニバッチの統計か移動統計どちらを使用するべき？
    # ミニバッチの統計を使う場合に備えてEstimatorのparams以外を一度固定
    # estimator.train, estimator.evaluateの呼び出し時にparamsとしてtrainingを与える
    Estimator = functools.partial(
        tf.estimator.Estimator,
        model_fn=lambda features, labels, mode, params: HATS(
            # =========================================================================================
            # feature extraction
            backbone_network=PyramidResNet(
                conv_param=Param(filters=64, kernel_size=[7, 7], strides=[2, 2]),
                pool_param=None,
                residual_params=[
                    Param(filters=64, strides=[2, 2], blocks=2),
                    Param(filters=128, strides=[2, 2], blocks=2),
                    Param(filters=256, strides=[2, 2], blocks=2),
                    Param(filters=512, strides=[2, 2], blocks=2),
                ],
                data_format=args.data_format
            ),
            # =========================================================================================
            # text detection
            attention_network=AttentionNetwork(
                conv_params=[
                    Param(filters=16, kernel_size=[3, 3], strides=[2, 2]),
                    Param(filters=16, kernel_size=[3, 3], strides=[2, 2]),
                ],
                rnn_params=[
                    Param(sequence_length=5, num_units=256),
                    Param(sequence_length=11, num_units=256),
                ],
                deconv_params=[
                    Param(filters=16, kernel_size=[3, 3], strides=[2, 2]),
                    Param(filters=16, kernel_size=[3, 3], strides=[2, 2]),
                ],
                data_format=args.data_format
            ),
            # =========================================================================================
            # text recognition
            num_units=[1024],
            num_classes=37,
            # =========================================================================================
            data_format=args.data_format,
            hyper_params=Param(
                attention_decay=0.0,
                learning_rate_fn=lambda global_step: tf.train.exponential_decay(
                    learning_rate=1e-3,
                    global_step=global_step,
                    decay_steps=25000,
                    decay_rate=1e-1,
                    staircase=True
                )
            )
        )(features, labels, mode, Param(params)),
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
        ),
        # resnetはchars74kで学習させた重みを初期値として用いる
        warm_start_from=tf.estimator.WarmStartSettings(
            ckpt_to_initialize_from=args.pretrained_model_dir,
            vars_to_warm_start=".*pyramid_resnet.*"
        )
    )

    if args.train:

        Estimator(params=dict(training=True)).train(
            input_fn=functools.partial(
                dataset.input_fn,
                filenames=args.train_filenames,
                batch_size=args.batch_size,
                num_epochs=None,
                shuffle=True,
                sequence_lengths=[5, 11],
                encoding="jpeg",
                image_size=[256, 256],
                data_format=args.data_format
            ),
            max_steps=args.max_steps,
            hooks=[
                # logging用のhook
                tf.train.LoggingTensorHook(
                    tensors=dict(
                        word_accuracy="accuracy",
                        edit_distance="distance"
                    ),
                    every_n_iter=100
                ),
                # validationのためのcustom hook
                # 内部でestimator.evaluateしている
                hooks.ValidationMonitorHook(
                    estimator=Estimator(params=dict(training=True)),
                    input_fn=functools.partial(
                        dataset.input_fn,
                        filenames=args.val_filenames,
                        batch_size=args.batch_size,
                        num_epochs=1,
                        shuffle=False,
                        sequence_lengths=[5, 11],
                        encoding="jpeg",
                        image_size=[256, 256],
                        data_format=args.data_format
                    ),
                    every_n_steps=1000,
                    steps=1000,
                    name="validation"
                )
            ]
        )

    if args.eval:

        eval_result = Estimator(params=dict(training=True)).evaluate(
            input_fn=functools.partial(
                dataset.input_fn,
                filenames=args.test_filenames,
                batch_size=args.batch_size,
                num_epochs=1,
                shuffle=False,
                sequence_lengths=[5, 11],
                encoding="jpeg",
                image_size=[256, 256],
                data_format=args.data_format
            ),
            steps=args.steps,
            name="test"
        )

        print("==================================================")
        tf.logging.info("test result")
        tf.logging.info(eval_result)
        print("==================================================")
