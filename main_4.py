import tensorflow as tf
import numpy as np
import sys
import argparse
import itertools
import math
import cv2
from attrdict import AttrDict
from dataset_2 import Dataset
from model_3 import Model
from networks.residual_network import ResidualNetwork
from networks.attention_network import AttentionNetwork

parser = argparse.ArgumentParser()
parser.add_argument("--model_dir", type=str, default="model_3", help="model directory")
parser.add_argument('--filenames', type=str, nargs="+", default=["train.tfrecord"], help="tfrecord filenames")
parser.add_argument("--num_epochs", type=int, default=10, help="number of training epochs")
parser.add_argument("--batch_size", type=int, default=128, help="batch size")
parser.add_argument("--buffer_size", type=int, default=7000000, help="buffer size to shuffle dataset")
parser.add_argument("--train", action="store_true", help="with training")
parser.add_argument("--eval", action="store_true", help="with evaluation")
parser.add_argument("--predict", action="store_true", help="with prediction")
parser.add_argument("--gpu", type=str, default="0,1,2", help="gpu id")
args = parser.parse_args()

tf.logging.set_verbosity(tf.logging.INFO)

sys.setrecursionlimit(10000)


def scale(input, input_min, input_max, output_min, output_max):
    return output_min + (input - input_min) / (input_max - input_min) * (output_max - output_min)


def search_bounding_box(image, threshold):

    if len(image.shape) == 3 and image.shape[-1] == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    max_value = 1.0 if np.issubdtype(image.dtype, np.floating) else 255
    binary = cv2.threshold(image, threshold, max_value, cv2.THRESH_BINARY)[1]
    flags = np.ones_like(binary, dtype=np.bool)
    h, w = binary.shape[:2]
    segments = []

    def depth_first_search(y, x):

        segments[-1].append((y, x))
        flags[y][x] = False

        for dy, dx in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
            if 0 <= y + dy < h and 0 <= x + dx < w:
                if flags[y + dy, x + dx] and binary[y + dy, x + dx]:
                    depth_first_search(y + dy, x + dx)

    for y in range(flags.shape[0]):
        for x in range(flags.shape[1]):
            if flags[y, x] and binary[y, x]:
                segments.append([])
                depth_first_search(y, x)

    bounding_boxes = [(lambda ls_1, ls_2: ((min(ls_1), min(ls_2)), (max(ls_1), max(ls_2))))(*zip(*segment)) for segment in segments]
    bounding_boxes = sorted(bounding_boxes, key=lambda box: abs(box[0][0] - box[1][0]) * abs(box[0][1] - box[1][1]))

    return bounding_boxes[-1]


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
                data_format="channels_last"
            ),
            num_units=128,
            num_classes=63,
            data_format="channels_last",
            accuracy_type=Model.AccuracyType.EDIT_DISTANCE,
            hyper_params=None
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
                image_size=[256, 256],
                data_format="channels_last",
                string_length=10
            ).get_next()
        )

    if args.eval:

        eval_results = classifier.evaluate(
            input_fn=lambda: Dataset(
                filenames=args.filenames,
                num_epochs=args.num_epochs,
                batch_size=args.batch_size,
                buffer_size=args.buffer_size,
                image_size=[256, 256],
                data_format="channels_last",
                string_length=10
            ).get_next()
        )

        print(eval_results)


if __name__ == "__main__":
    tf.app.run()
