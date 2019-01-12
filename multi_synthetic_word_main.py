import tensorflow as tf
import numpy as np
import argparse
from attrdict import AttrDict
from dataset import Dataset
from model import Model
from networks.residual_network import ResidualNetwork
from networks.attention_network import AttentionNetwork

parser = argparse.ArgumentParser()
parser.add_argument("--model_dir", type=str, default="multi_synthetic_word_acnn_model", help="model directory")
parser.add_argument('--filenames', type=str, nargs="+", default=["multi_synthetic_word_train.tfrecord"], help="tfrecord filenames")
parser.add_argument("--num_epochs", type=int, default=10, help="number of training epochs")
parser.add_argument("--batch_size", type=int, default=128, help="batch size")
parser.add_argument("--buffer_size", type=int, default=900000, help="buffer size to shuffle dataset")
parser.add_argument("--data_format", type=str, default="channels_first", help="data format")
parser.add_argument("--train", action="store_true", help="with training")
parser.add_argument("--eval", action="store_true", help="with evaluation")
parser.add_argument("--predict", action="store_true", help="with prediction")
parser.add_argument("--gpu", type=str, default="0,1,2", help="gpu id")
args = parser.parse_args()

tf.logging.set_verbosity(tf.logging.INFO)


def get_learning_rate_fn_with_decay(base_learning_rate, batch_size, batch_denom,
                                    num_data, boundary_epochs, decay_rates):

    initial_learning_rate = base_learning_rate * batch_size / batch_denom
    batches_per_epoch = num_data / batch_size
    boundaries = [int(batches_per_epoch * boundary_epoch) for boundary_epoch in boundary_epochs]
    values = [initial_learning_rate * decay_rate for decay_rate in decay_rates]

    return lambda global_step: tf.train.piecewise_constant(global_step, boundaries, values)


def main(unused_argv):

    classifier = tf.estimator.Estimator(
        model_fn=Model(
            convolutional_network=ResidualNetwork(
                conv_param=AttrDict(filters=64, kernel_size=[7, 7], strides=[2, 2]),
                pool_param=None,
                residual_params=[
                    AttrDict(filters=64, strides=[2, 2], blocks=2),
                    AttrDict(filters=128, strides=[2, 2], blocks=2),
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
                rnn_params=[
                    AttrDict(sequence_length=5, num_units=256),
                    AttrDict(sequence_length=10, num_units=256),
                ],
                data_format=args.data_format
            ),
            num_classes=63,
            data_format=args.data_format,
            hyper_params=AttrDict(
                attention_map_decay=1e-3,
                weight_decay=1e-4,
                loss_filter_fn=lambda name: "batch_normalization" not in name,
                learning_rate_fn=get_learning_rate_fn_with_decay(
                    base_learning_rate=0.128,
                    batch_size=args.batch_size,
                    batch_denom=256,
                    num_data=args.buffer_size,
                    boundary_epochs=[2, 4, 6, 8],
                    decay_rates=[1e-0, 1e-1, 1e-2, 1e-3, 1e-4]
                ),
                momentum=0.9,
                # learning_rate=0.001,
                # beta1=0.9,
                # beta2=0.999,
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

        classifier.train(
            input_fn=lambda: Dataset(
                filenames=args.filenames,
                num_epochs=args.num_epochs,
                batch_size=args.batch_size,
                buffer_size=args.buffer_size,
                sequence_lengths=[5, 10],
                image_size=[256, 256],
                data_format=args.data_format
            ).get_next()
        )

    if args.eval:

        eval_results = classifier.evaluate(
            input_fn=lambda: Dataset(
                filenames=args.filenames,
                num_epochs=args.num_epochs,
                batch_size=args.batch_size,
                buffer_size=args.buffer_size,
                sequence_lengths=[5, 10],
                image_size=[256, 256],
                data_format=args.data_format
            ).get_next()
        )

        print(eval_results)


if __name__ == "__main__":
    tf.app.run()
