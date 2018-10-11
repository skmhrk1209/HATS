import tensorflow as tf
import numpy as np
import argparse
from utils.attr_dict import AttrDict
from data.imagenet import Dataset
from models.acnn import Model
from networks.residual_network import ResidualNetwork
from networks.attention_network import AttentionNetwork

parser = argparse.ArgumentParser()
parser.add_argument("--model_dir", type=str, default="imagenet_acnn_model", help="model directory")
parser.add_argument('--filenames', type=str, nargs="+", default=["train.tfrecord"], help="tfrecord filenames")
parser.add_argument("--num_epochs", type=int, default=100, help="number of training epochs")
parser.add_argument("--batch_size", type=int, default=100, help="batch size")
parser.add_argument("--buffer_size", type=int, default=1000000, help="buffer size to shuffle dataset")
parser.add_argument('--data_format', type=str, choices=["channels_first", "channels_last"], default="channels_last", help="data_format")
parser.add_argument('--train', action="store_true", help="with training")
parser.add_argument('--eval', action="store_true", help="with evaluation")
parser.add_argument('--predict', action="store_true", help="with prediction")
parser.add_argument('--gpu', type=str, default="0", help="gpu id")
args = parser.parse_args()

tf.logging.set_verbosity(tf.logging.INFO)


def main(unused_argv):

    imagenet_classifier = tf.estimator.Estimator(
        model_fn=Model(
            convolutional_network=ResidualNetwork(
                conv_param=AttrDict(filters=64, kernel_size=[7, 7], strides=[1, 1]),
                pool_param=None,
                residual_params=[
                    AttrDict(filters=64, strides=[1, 1], blocks=3),
                    AttrDict(filters=128, strides=[2, 2], blocks=4),
                    AttrDict(filters=256, strides=[2, 2], blocks=6),
                    AttrDict(filters=512, strides=[2, 2], blocks=3),
                ],
                num_classes=None,
                data_format=args.data_format
            ),
            attention_network=AttentionNetwork(
                conv_params=[
                    AttrDict(filters=4, kernel_size=[9, 9], strides=[2, 2]),
                    AttrDict(filters=4, kernel_size=[9, 9], strides=[2, 2]),
                ],
                deconv_params=[
                    AttrDict(filters=16, kernel_size=[3, 3], strides=[2, 2]),
                    AttrDict(filters=16, kernel_size=[3, 3], strides=[2, 2]),
                ],
                bottleneck_units=128,
                data_format=args.data_format
            ),
            num_classes=1000,
            data_format=args.data_format,
            hyper_parameters=AttrDict(
                weight_decay=1e-4,
                attention_decay=1e-6,
                momentum=0.9,
                learning_rate_fn=lambda global_step: tf.cond(
                    pred=global_step < 50000,
                    true_fn=lambda: 0.1 * tf.cast(global_step, tf.float32) / 50000.0,
                    false_fn=lambda: tf.train.piecewise_constant(
                        x=global_step,
                        boundaries=[300000, 600000, 800000, 900000],
                        values=[0.1, 0.01, 0.001, 0.0001, 0.00001]
                    )
                )
            )
        ),
        model_dir=args.model_dir,
        config=tf.estimator.RunConfig().replace(
            session_config=tf.ConfigProto(
                gpu_options=tf.GPUOptions(
                    visible_device_list=args.gpu,
                    allow_growth=True
                )
            )
        )
    )

    if args.train:

        imagenet_classifier.train(
            input_fn=lambda: Dataset(
                filenames=args.filenames,
                num_epochs=args.num_epochs,
                batch_size=args.batch_size,
                buffer_size=args.buffer_size
            ).get_next(),
            hooks=[
                tf.train.LoggingTensorHook(
                    tensors={"softmax": "softmax"},
                    every_n_iter=100
                )
            ]
        )

    if args.eval:

        eval_results = imagenet_classifier.evaluate(
            input_fn=lambda: Dataset(
                filenames=args.filenames,
                num_epochs=args.num_epochs,
                batch_size=args.batch_size,
                buffer_size=args.buffer_size
            ).get_next()
        )

        print(eval_results)


if __name__ == "__main__":
    tf.app.run()
