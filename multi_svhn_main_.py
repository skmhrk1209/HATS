import tensorflow as tf
import numpy as np
import argparse
import itertools
import seaborn
import matplotlib.pyplot as plt
from utils.attr_dict import AttrDict
from data.multi_svhn import Dataset
from models.acnn_ import Model
from networks.residual_network import ResidualNetwork
from networks.hierarchical_recurrent_attention_network import HierarchicalRecurrentAttentionNetwork

parser = argparse.ArgumentParser()
parser.add_argument("--model_dir", type=str, default="multi_svhn_acnn_model_", help="model directory")
parser.add_argument('--filenames', type=str, nargs="+", default=["multi_svhn_train.tfrecord"], help="tfrecord filenames")
parser.add_argument("--num_epochs", type=int, default=100, help="number of training epochs")
parser.add_argument("--batch_size", type=int, default=128, help="batch size")
parser.add_argument("--buffer_size", type=int, default=100000, help="buffer size to shuffle dataset")
parser.add_argument("--data_format", type=str, choices=["channels_first", "channels_last"], default="channels_last", help="data_format")
parser.add_argument("--train", action="store_true", help="with training")
parser.add_argument("--eval", action="store_true", help="with evaluation")
parser.add_argument("--predict", action="store_true", help="with prediction")
parser.add_argument("--gpu", type=str, default="0", help="gpu id")
args = parser.parse_args()

tf.logging.set_verbosity(tf.logging.INFO)


def main(unused_argv):

    imagenet_classifier = tf.estimator.Estimator(
        model_fn=Model(
            convolutional_network=ResidualNetwork(
                conv_param=AttrDict(filters=32, kernel_size=[3, 3], strides=[1, 1]),
                pool_param=None,
                residual_params=[
                    AttrDict(filters=32, strides=[2, 2], blocks=2),
                    AttrDict(filters=64, strides=[2, 2], blocks=2),
                    AttrDict(filters=128, strides=[1, 1], blocks=2),
                    AttrDict(filters=256, strides=[1, 1], blocks=2),
                ],
                num_classes=None,
                data_format=args.data_format
            ),
            recurrent_attention_network=HierarchicalRecurrentAttentionNetwork(
                conv_params=[
                    AttrDict(filters=4, kernel_size=[9, 9], strides=[2, 2]),
                    AttrDict(filters=4, kernel_size=[9, 9], strides=[2, 2]),
                ],
                deconv_params=[
                    AttrDict(filters=16, kernel_size=[3, 3], strides=[2, 2]),
                    AttrDict(filters=16, kernel_size=[3, 3], strides=[2, 2]),
                ],
                global_bottleneck_units=32,
                local_bottleneck_units=16,
                sequence_length=4,
                digits_length=4,
                data_format=args.data_format
            ),
            num_classes=11,
            data_format=args.data_format,
            hyper_params=AttrDict(
                cross_entropy_decay=1e-0,
                attention_map_decay=1e-0,#1e-1,
                total_variation_decay=1e-4#1e-6
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
                buffer_size=args.buffer_size,
                data_format=args.data_format,
                image_size=[128, 128],
                digits_length=4,
                sequence_length=4
            ).get_next()
        )

    if args.eval:

        eval_results = imagenet_classifier.evaluate(
            input_fn=lambda: Dataset(
                filenames=args.filenames,
                num_epochs=args.num_epochs,
                batch_size=args.batch_size,
                buffer_size=args.buffer_size,
                data_format=args.data_format,
                image_size=[128, 128],
                digits_length=4,
                sequence_length=4
            ).get_next()
        )

        print(eval_results)

    if args.predict:

        predict_results = imagenet_classifier.predict(
            input_fn=lambda: Dataset(
                filenames=args.filenames,
                num_epochs=args.num_epochs,
                batch_size=args.batch_size,
                buffer_size=args.buffer_size,
                data_format=args.data_format,
                image_size=[128, 128],
                digits_length=4,
                sequence_length=4
            ).get_next()
        )

        for i, predict_result in enumerate(itertools.islice(predict_results, 10)):

            reduced_attention_map = predict_result["reduced_attention_maps"]
            seaborn.heatmap(reduced_attention_map.reshape([32, 32]))

            plt.savefig("output/reduced_attention_map_{}.png".format(i))


if __name__ == "__main__":
    tf.app.run()
