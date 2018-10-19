import tensorflow as tf
import numpy as np
import itertools
import argparse
import os
import glob

parser = argparse.ArgumentParser()
parser.add_argument("--filename", type=str, required=True, help="tfrecord filename")
parser.add_argument("--directory", type=str, required=True, help="path to data directory")
args = parser.parse_args()

with tf.python_io.TFRecordWriter(args.filename) as writer:

    max_length = 4

    for file in glob.glob(os.path.join(args.directory, "*")):

        label = [
            digit for label in os.path.splitext(os.path.basename(file))[0].split("-")[1:]
            for digit in np.pad([int(digit) for digit in label], [0, max_length - len(label)], "constant", constant_values=10)
        ]

        writer.write(
            record=tf.train.Example(
                features=tf.train.Features(
                    feature={
                        "path": tf.train.Feature(
                            bytes_list=tf.train.BytesList(
                                value=[file.encode("utf-8")]
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
