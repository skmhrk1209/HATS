import tensorflow as tf
import numpy as np
import argparse
import os
import glob

parser = argparse.ArgumentParser()
parser.add_argument("--filename", type=str, required=True, help="tfrecord filename")
parser.add_argument("--directory", type=str, required=True, help="path to data directory")
parser.add_argument("--string_length", type=int, default=10, help="max string length")
args = parser.parse_args()

with tf.python_io.TFRecordWriter(args.filename) as writer:

    for file in glob.glob(os.path.join(args.directory, "*")):

        def convert(char):
            return ord(char) - 48 if char <= "9" else ord(char) - 55 if char <= "Z" else ord(char) - 61

        string = os.path.splitext(os.path.basename(file))[0].split("_")[1]

        label = np.pad(
            array=string,
            pad_width=[[0, args.string_length - len(string)]],
            mode="constant",
            constant_values=62
        ).astype(np.int32).tolist()

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
