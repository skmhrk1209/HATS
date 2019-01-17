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
from models.hats import HATS
from networks.han import HAN
from networks.resnet import ResNet
from algorithms import *

parser = argparse.ArgumentParser()
parser.add_argument("--model_dir", type=str, default="word_recognition_hats_model", help="model directory")
parser.add_argument("--pretrained_model_dir", type=str, default="", help="pretrained model directory")
parser.add_argument('--filenames', type=str, nargs="+", default=["word_recognition_train.tfrecord"], help="tfrecord filenames")
parser.add_argument("--num_epochs", type=int, default=1000, help="number of training epochs")
parser.add_argument("--batch_size", type=int, default=128, help="batch size")
parser.add_argument("--buffer_size", type=int, default=4468, help="buffer size to shuffle dataset")
parser.add_argument("--data_format", type=str, default="channels_first", help="data format")
parser.add_argument("--train", action="store_true", help="with training")
parser.add_argument("--eval", action="store_true", help="with evaluation")
parser.add_argument("--predict", action="store_true", help="with prediction")
parser.add_argument("--gpu", type=str, default="0,1,2", help="gpu id")
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
                num_classes=None,
                data_format=args.data_format
            ),
            hierarchical_attention_network=HAN(
                conv_params=[
                    AttrDict(filters=16, kernel_size=[9, 9], strides=[2, 2]),
                    AttrDict(filters=16, kernel_size=[9, 9], strides=[2, 2]),
                ],
                deconv_params=[
                    AttrDict(filters=16, kernel_size=[3, 3], strides=[2, 2]),
                    AttrDict(filters=16, kernel_size=[3, 3], strides=[2, 2]),
                ],
                rnn_params=[
                    AttrDict(sequence_length=21, num_units=256)
                ],
                data_format=args.data_format
            ),
            num_classes=96,
            data_format=args.data_format,
            hyper_params=AttrDict(
                attention_decay=1e-4,
                learning_rate=0.001,
                beta1=0.9,
                beta2=0.999
            ),
            pretrained_model_dir=args.pretrained_model_dir
        )(features, labels, mode),
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
            input_fn=Dataset(
                filenames=args.filenames,
                num_epochs=args.num_epochs,
                batch_size=args.batch_size,
                buffer_size=args.buffer_size,
                sequence_lengths=[21],
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
                sequence_lengths=[21],
                image_size=[256, 256],
                data_format=args.data_format
            )
        )

        print(eval_results)

    if args.predict:

        filenames = glob.glob("evaluation_dataset/*.png")

        predict_results = classifier.predict(
            input_fn=lambda: tf.data.Dataset.from_tensor_slices(filenames).map(compose(
                functools.partial(tf.read_file),
                functools.partial(tf.image.decode_png, channels=3),
                functools.partial(tf.image.convert_image_dtype, dtype=tf.float32),
                functools.partial(tf.image.resize_images, size=[256, 256]),
                functools.partial(tf.transpose, perm=[2, 0, 1] if args.data_format == "channels_first" else [0, 1, 2])
            )).batch(args.batch_size).make_one_shot_iterator().get_next()
        )

        with open("result.txt", "w") as f:

            for filename, predict_result in zip(filenames, predict_results):

                prediction = "".join(map(chr, predict_result["predictions"][predict_result["predictions"] < 95] + 32))
                f.write('{}, "{}"\n'.format(os.path.basename(filename), prediction))


if __name__ == "__main__":
    tf.app.run()
