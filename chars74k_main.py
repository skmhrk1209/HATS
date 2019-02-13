# =============================================================
# dataset details
# dataset: chars74k
# download: http://www.ee.surrey.ac.uk/CVSSP/demos/chars74k/
# train: 11252
# test: 1251
# classes: 37 [0-9A-Z](case-insensitive)
# accuracy: 0.894
# pretrained model: chars74k classifier
# max steps: 10000 batch size: 128
# =============================================================

import tensorflow as tf
import argparse
import functools
import dataset
from models.hats import HATS
from networks.attention_network import AttentionNetwork
from networks.pyramid_resnet import PyramidResNet
from attrdict import AttrDict as Param

parser = argparse.ArgumentParser()
parser.add_argument("--model_dir", type=str, default="chars74k_hats_model", help="model directory")
parser.add_argument("--pretrained_model_dir", type=str, default="", help="pretrained model directory")
parser.add_argument('--train_filenames', type=str, nargs="+", default=["chars74k_train.tfrecord"], help="tfrecords for training")
parser.add_argument('--val_filenames', type=str, nargs="+", default=["chars74k_val.tfrecord"], help="tfrecords for validation")
parser.add_argument('--test_filenames', type=str, nargs="+", default=["chars74k_test.tfrecord"], help="tfrecords for test")
parser.add_argument("--batch_size", type=int, default=100, help="batch size")
parser.add_argument("--random_seed", type=int, default=1209, help="random seed")
parser.add_argument("--data_format", type=str, default="channels_first", help="data format")
parser.add_argument("--max_steps", type=int, default=10000, help="maximum number of training steps")
parser.add_argument("--steps", type=int, default=None, help="number of evaluation steps")
parser.add_argument('--train', action="store_true", help="with training")
parser.add_argument('--eval', action="store_true", help="with evaluation")
parser.add_argument("--gpu", type=str, default="0", help="gpu id")
args = parser.parse_args()

tf.logging.set_verbosity(tf.logging.INFO)


if __name__ == "__main__":

    estimator = tf.estimator.Estimator(
        model_fn=lambda features, labels, mode: Classifier(
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
            num_classes=37,
            data_format=args.data_format,
            hyper_params=Param(
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
        ),
        warm_start_from=tf.estimator.WarmStartSettings(
            ckpt_to_initialize_from=args.pretrained_model_dir,
            vars_to_warm_start=".*pyramid_resnet.*"
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
                sequence_lengths=[],
                encoding="png",
                image_size=[128, 128],
                data_format=args.data_format
            ),
            max_steps=args.max_steps
        )

    if args.eval:

        print(estimator.evaluate(
            input_fn=functools.partial(
                dataset.input_fn,
                filenames=args.test_filenames,
                batch_size=args.batch_size,
                num_epochs=1,
                shuffle=False,
                sequence_lengths=[],
                encoding="png",
                image_size=[128, 128],
                data_format=args.data_format
            ),
            steps=args.steps
        ))
