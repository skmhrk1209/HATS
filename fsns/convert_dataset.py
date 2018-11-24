import tensorflow as tf
import numpy as np
import argparse
import os
import glob

parser = argparse.ArgumentParser()
parser.add_argument("--filename", type=str, required=True, help="tfrecord filename")
parser.add_argument("--directory", type=str, required=True, help="path to data directory")
parser.add_argument("--sequence_length", type=int, default=9, help="max sequence length")
parser.add_argument("--string_length", type=int, default=35, help="max string length")
parser.add_argument("--class_ids", type=str, default="class_ids.txt", help="class ids file")
args = parser.parse_args()

with tf.python_io.TFRecordWriter(args.filename) as writer:

    class_ids = {}

    with open(args.class_ids, "r") as f:
        for line in f:
            class_id, char = line.split()
            class_ids[char] = class_id

    null_class_id = max(class_ids.values()) + 1

    for file in glob.glob(os.path.join(args.directory, "*")):

        strings = os.path.splitext(os.path.basename(file))[0].split("_")[1:]

        label = np.pad(
            array=[
                np.pad(
                    array=[class_ids(char) for char in string],
                    pad_width=[[0, args.string_length - len(string)]],
                    mode="constant",
                    constant_values=null_class_id
                ) for string in strings
            ],
            pad_width=[[0, args.sequence_length - len(strings)], [0, 0]],
            mode="constant",
            constant_values=null_class_id
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
