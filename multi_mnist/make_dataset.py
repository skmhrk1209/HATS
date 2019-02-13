import tensorflow as tf
import numpy as np
import skimage
import sys
import os
import itertools
import argparse
import random
from tqdm import trange
from numba import jit
from shapely.geometry import box

# TODO: warning with skimage.io.imsave
# like that UserWarning: xxxx.png is a low contrast image
import warnings
warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser()
parser.add_argument("--train_num_digits", type=int, default=4, help="number of digits contained in a train instance")
parser.add_argument("--test_num_digits", type=int, default=4, help="number of digits contained in a test instance")
parser.add_argument("--train_num_instances", type=int, default=60000, help="number of train instances")
parser.add_argument("--test_num_instances", type=int, default=10000, help="number of test instances")
parser.add_argument("--train_image_size", type=int, nargs="+", default=[256, 256], help="image_size of a train instance")
parser.add_argument("--test_image_size", type=int, nargs="+", default=[256, 256], help="image_size of a test instance")
parser.add_argument("--train_data_dir", type=str, default="train", help="train data directory")
parser.add_argument("--test_data_dir", type=str, default="test", help="test data directory")
args = parser.parse_args()


def main(images, labels, num_digits, num_instances, image_size, data_dir):

    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    for i in trange(num_instances):

        num_samples = random.randint(1, num_digits)
        samples = random.sample(list(zip(images, labels)), num_samples)

        sample_images = []
        sample_labels = []
        sample_bboxes = []

        for sample_image, sample_label in samples:

            while True:
                y = random.randint(0, image_size[0] - 28)
                x = random.randint(0, image_size[1] - 28)
                sample_bbox = (y, y + 28, x, x + 28)

                if all([box(*sample_bbox_).disjoint(box(*sample_bbox)) for sample_bbox_ in sample_bboxes]):
                    sample_image = np.pad(sample_image, [[y, image_size[0] - y - 28], [x, image_size[1] - x - 28]], "constant")
                    break

            sample_images.append(sample_image)
            sample_labels.append(sample_label)
            sample_bboxes.append(sample_bbox)

        sample_labels = [sample_label for _, sample_label in sorted(zip(sample_bboxes, sample_labels), key=lambda item: (item[0][0], item[0][2]))]
        multi_image = sum(sample_images)
        multi_label = "".join(map(str, sample_labels))
        skimage.io.imsave(os.path.join(data_dir, "{}.jpg".format(multi_label)), multi_image)


if __name__ == "__main__":

    mnist = tf.keras.datasets.mnist
    train_data, test_data = mnist.load_data()
    main(*train_data, args.train_num_digits, args.train_num_instances, args.train_image_size, args.train_data_dir)
    main(*test_data, args.test_num_digits, args.test_num_instances, args.test_image_size, args.test_data_dir)
