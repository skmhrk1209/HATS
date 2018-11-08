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
    string_length = 10

    for file in glob.glob(os.path.join(args.directory, "*")):

        def convert(char):
            return ord(char) - 48 if char <= "9" else ord(char) - 55 if char <= "Z" else ord(char) - 61

        strings = os.path.splitext(os.path.basename(file))[0].split("_")[1:]

        if len(strings) > 10: print(len(strings))

        label = np.pad(
            array=[
                np.pad(
                    array=[convert(char) for char in string],
                    pad_width=[[0, string_length - len(string)]],
                    mode="constant",
                    constant_values=62
                ) for string in strings
            ],
            pad_width=[[0, sequence_length - len(strings)]],
            mode="constant",
            constant_values=62
        ).astype(np.int32)

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
                                value=label.reshape([-1]).tolist()
                            )
                        )
                    }
                )
            ).SerializeToString()
        )
