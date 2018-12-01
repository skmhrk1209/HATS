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

    class_ids = {}

    for i in range(ord("0"), ord("z") + 1):

        if ord("0") <= i <= ord("9"):
            class_ids[chr(i)] = i - ord("0")
        elif ord("A") <= i <= ord("Z"):
            class_ids[chr(i)] = i - ord("A") + class_ids["9"] + 1
        elif ord("a") <= i <= ord("z"):
            class_ids[chr(i)] = i - ord("a") + class_ids["Z"] + 1

    class_ids[""] = max(class_ids.values()) + 1

    for file in glob.glob(os.path.join(args.directory, "*")):

        string = os.path.splitext(os.path.basename(file))[0].split("_")[1]

        label = np.pad(
            array=[class_ids[char] for char in string],
            pad_width=[[0, args.string_length - len(string)]],
            mode="constant",
            constant_values=class_ids[""]
        )

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
                                value=label.astype(np.int32).reshape([-1]).tolist()
                            )
                        )
                    }
                )
            ).SerializeToString()
        )
