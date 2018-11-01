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

    sequence_length = 4
    string_length = 24

    for file in glob.glob(os.path.join(args.directory, "*")):

        def convert(c):
            return ord(c) - 48 if c <= "9" else ord(c) - 55 if c <= "Z" else ord(c) - 61

        labels = [[convert(c) for c in label] for label in os.path.splitext(os.path.basename(file))[0].split("_")[1:]]
        labels = [np.pad(label, [[0, string_length - len(label)]], "constant", constant_values=62) for label in labels]
        labels = np.pad(labels, [[0, sequence_length - len(labels)], [0, 0]], "constant", constant_values=62)
        labels = [c for label in labels for c in label]

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
