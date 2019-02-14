# =============================================================
# dataset details
# dataset: synth90k
# download: http://www.robots.ox.ac.uk/~vgg/data/text/
# train: 7224612
# val: 802734
# test: 891927
# max num chars: 23
# classes: 37 [0-9A-Z](case-insensitive)
# word accuracy:
# edit distance:
# pretrained model: chars74k classifier
# max steps: 50000 batch size: 128
# =============================================================

import tensorflow as tf
import argparse
import functools
import dataset
import hooks
from models.hats import HATS
from networks.attention_network import AttentionNetwork
from networks.pyramid_resnet import PyramidResNet
from attrdict import AttrDict as Param

parser = argparse.ArgumentParser()
parser.add_argument("--model_dir", type=str, default="synth90k_hats_model", help="model directory")
parser.add_argument("--pretrained_model_dir", type=str, default="chars74k_classifier", help="pretrained model directory")
parser.add_argument('--train_filenames', type=str, nargs="+", default=["synth90k_train.tfrecord"], help="tfrecords for training")
parser.add_argument('--val_filenames', type=str, nargs="+", default=["synth90k_val.tfrecord"], help="tfrecords for validation")
parser.add_argument('--test_filenames', type=str, nargs="+", default=["synth90k_test.tfrecord"], help="tfrecords for test")
parser.add_argument("--batch_size", type=int, default=100, help="batch size")
parser.add_argument("--random_seed", type=int, default=1209, help="random seed")
parser.add_argument("--data_format", type=str, default="channels_first", help="data format")
parser.add_argument("--max_steps", type=int, default=50000, help="maximum number of training steps")
parser.add_argument("--steps", type=int, default=1000, help="number of validation steps")
parser.add_argument('--train', action="store_true", help="with training")
parser.add_argument('--eval', action="store_true", help="with evaluation")
parser.add_argument("--gpu", type=str, default="0", help="gpu id")
args = parser.parse_args()

tf.logging.set_verbosity(tf.logging.INFO)


if __name__ == "__main__":

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
                    Param(filters=8, kernel_size=[3, 3], strides=[2, 2]),
                    Param(filters=8, kernel_size=[3, 3], strides=[2, 2]),
                ],
                rnn_params=[
                    Param(sequence_length=24, num_units=256),
                ],
                deconv_params=[
                    Param(filters=8, kernel_size=[3, 3], strides=[2, 2]),
                    Param(filters=8, kernel_size=[3, 3], strides=[2, 2]),
                ],
                data_format=args.data_format
            ),
            # =========================================================================================
            # text recognition
            num_units=[],
            num_classes=37,
            # =========================================================================================
            data_format=args.data_format,
            hyper_params=Param(
                # あんまり効果的ではない
                attention_decay=0.0,
                # 最適なepsilonが[1.0, 0.1]とかの時もあるらしい
                # 意味不明
                optimizer=tf.train.AdamOptimizer(
                    learning_rate=1e-3,
                    beta1=0.9,
                    beta2=0.999,
                    epsilon=1e-8
                )
            )
        )(features, labels, mode, params),
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

        Estimator(params=Param(training=True)).train(
            input_fn=functools.partial(
                dataset.input_fn,
                filenames=args.train_filenames,
                batch_size=args.batch_size,
                num_epochs=None,
                shuffle=True,
                sequence_lengths=[24],
                encoding="jpeg",
                image_size=[256, 256],
                data_format=args.data_format
            ),
            max_steps=args.max_steps,
            hooks=[
                # logging用のhook
                tf.train.LoggingTensorHook(
                    tensors={"word_accuracy": "word_accuracy"},
                    every_n_iter=100
                ),
                # validationのためのcustom hook
                # session.runの後にestimator.evaluateしてるだけ
                hooks.ValidationHook(
                    estimator=Estimator(params=Param(training=True)),
                    input_fn=functools.partial(
                        dataset.input_fn,
                        filenames=args.val_filenames,
                        batch_size=args.batch_size,
                        num_epochs=1,
                        shuffle=False,
                        sequence_lengths=[24],
                        encoding="jpeg",
                        image_size=[256, 256],
                        data_format=args.data_format
                    ),
                    every_n_steps=1000,
                    steps=args.steps,
                    name="validation"
                )
            ]
        )

    if args.eval:

        print(Estimator(params=Param(training=False)).evaluate(
            input_fn=functools.partial(
                dataset.input_fn,
                filenames=args.test_filenames,
                batch_size=args.batch_size,
                num_epochs=1,
                shuffle=False,
                sequence_lengths=[24],
                encoding="jpeg",
                image_size=[256, 256],
                data_format=args.data_format
            ),
            name="test"
        ))
