import tensorflow as tf
import numpy as np
import argparse
import glob
import os
from tqdm import *


parser = argparse.ArgumentParser()
parser.add_argument("--input_directory", type=str, default="multi_mnist/train", help="input multi-mnist directory")
parser.add_argument("--output_filename", type=str, default="multi_mnist/train.tfrecord", help="output tfrecord filename")
parser.add_argument("--num_digits", type=int, default=5, help="number of digits contained in a instance (include blank)")
args = parser.parse_args()


def pad(sequence, sequence_length, value):
    while len(sequence) < sequence_length:
        sequence.append(value)
    return sequence


def main(input_directory, output_filename, num_digits):

    with tf.python_io.TFRecordWriter(output_filename) as writer:

        for filename in glob.glob(os.path.join(input_directory, "*.jpg")):

            label = list(map(int, list(os.path.splitext(os.path.basename(filename))[0])))
            label = pad(label, num_digits, 10)

            writer.write(
                record=tf.train.Example(
                    features=tf.train.Features(
                        feature={
                            "path": tf.train.Feature(
                                bytes_list=tf.train.BytesList(
                                    value=[filename.encode("utf-8")]
                                )
                            ),
                            "label": tf.train.Feature(
                                int64_list=tf.train.Int64List(
                                    value=label
                                )
                            )
                        }
                    )
                ).SerializeToString()
            )


if __name__ == "__main__":

    main(*sys.argv[1:3], *list(map(int, sys.argv[3:])))
