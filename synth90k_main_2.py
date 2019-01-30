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


def main(unused_argv):

    classifier = tf.estimator.Estimator(
        model_fn=lambda features, labels, mode: HATS(
            backbone_network=lambda inputs, training: tf.image.resize_image_with_crop_or_pad(hub.Module(
                spec="https://tfhub.dev/google/imagenet/resnet_v1_50/feature_vector/1",
                trainable=True,
                name="resnet",
                tags={"train"} if training else None
            )(
                inputs=dict(images=inputs),
                signature="image_feature_vector",
                as_dict=True
            )["resnet_v1_50/block4"], *[8, 8]),
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
                attention_decay=1e-5,
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
                image_size=[224, 224],
                data_format=args.data_format
            ),
            steps=args.steps,
            max_steps=args.max_steps,
            hooks=[
                tf.train.LoggingTensorHook(
                    tensors={
                        "edit_distance": "edit_distance",
                        "sequence_accuracy": "sequence_accuracy"
                    },
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
                image_size=[224, 224],
                data_format=args.data_format
            )
        )

        print(eval_results)


if __name__ == "__main__":
    tf.app.run()