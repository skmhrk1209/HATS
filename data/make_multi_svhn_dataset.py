import tensorflow as tf
import numpy as np
import argparse
import os
import glob

parser = argparse.ArgumentParser()
parser.add_argument("--filename", type=str, required=True, help="tfrecord filename")
parser.add_argument("--directory", type=str, required=True, help="path to data directory")
args = parser.parse_args()

with tf.python_io.TFRecordWriter(args.filename) as writer:

    digits_length = 4
    sequence_length = 4

    for file in glob.glob(os.path.join(args.directory, "*")):

        labels = [[int(digit) for digit in label] for label in os.path.splitext(os.path.basename(file))[0].split("-")[1:]]
        labels = [np.pad(label, [0, digits_length - len(label)], "constant", constant_values=10) for label in labels]
        labels = np.pad(labels, [0, sequence_length - len(labels)], "constant", constant_values=[10] * digits_length)
        labels = [digit for label in labels for digit in label]

        writer.write(
            record=tf.train.Example(
                features=tf.train.Features(
                    feature={
                        "path": tf.train.Feature(
                            bytes_list=tf.train.BytesList(
                                value=[file.encode("utf-8")]
                            )
                        ),
                        "labels": tf.train.Feature(
                            int64_list=tf.train.Int64List(
                                value=labels
                            )
                        )
                    }
                )
            ).SerializeToString()
        )
