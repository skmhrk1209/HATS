import tensorflow as tf
import numpy as np
import argparse
import os
import glob

parser = argparse.ArgumentParser()
parser.add_argument("--filename", type=str, required=True, help="tfrecord filename")
parser.add_argument("--directory", type=str, required=True, help="path to data directory")
parser.add_argument("--sequence_length", type=int, default=4, help="max sequence length")
parser.add_argument("--string_length", type=int, default=10, help="max string length")
args = parser.parse_args()

with tf.python_io.TFRecordWriter(args.filename) as writer:

    for file in glob.glob(os.path.join(args.directory, "*")):

        def to_label(char):
            return (ord(char) - ord("0") if char <= "9" else
                    ord(char) - ord("A") + (to_label("9") + 1) if char <= "Z" else
                    ord(char) - ord("a") + (to_label("Z") + 1) if char <= "z" else to_label("z") + 1)

        def to_char(label):
            return (chr(label + ord("0")) if label <= to_label("9") else
                    chr(label + ord("A") - (to_label("9") + 1)) if label <= to_label("Z") else
                    chr(label + ord("a") - (to_label("Z") + 1)) if label <= to_label("z") else "")

        strings = os.path.splitext(os.path.basename(file))[0].split("_")[1:]

        label = np.pad(
            array=[np.pad(
                array=[to_label(char) for char in string],
                pad_width=[[0, args.string_length - len(string)]],
                mode="constant",
                constant_values=62
            ) for string in strings],
            pad_width=[[0, args.sequence_length - len(strings)], [0, 0]],
            mode="constant",
            constant_values=62
        ).astype(np.int32).reshape([-1]).tolist()

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
