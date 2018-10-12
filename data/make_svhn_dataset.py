import tensorflow as tf
import numpy as np
import argparse
import os
import glob
from digit_struct import DigitStruct
import cv2

parser = argparse.ArgumentParser()
parser.add_argument("--filename", type=str, required=True, help="tfrecord filename")
parser.add_argument("--directory", type=str, required=True, help="path to data directory")
args = parser.parse_args()

with tf.python_io.TFRecordWriter(args.filename) as writer:

    digit_struct = DigitStruct(os.path.join(args.directory, "digitStruct.mat"))

    max_length = 4

    for struct in digit_struct.get_all_digit_structs():

        length = len(struct["label"])

        if length > max_length:
            continue

        name = os.path.join(args.directory, struct["name"])
        label = np.asarray(struct["label"], dtype=np.int32)
        top = np.asarray(struct["top"], dtype=np.int32)
        left = np.asarray(struct["left"], dtype=np.int32)
        height = np.asarray(struct["height"], dtype=np.int32)
        width = np.asarray(struct["width"], dtype=np.int32)
        bottom = top + height
        right = left + width

        label = np.pad(
            array=label % 10,
            pad_width=[0, max_length - length],
            mode="constant",
            constant_values=10
        )

        writer.write(
            record=tf.train.Example(
                features=tf.train.Features(
                    feature={
                        "path": tf.train.Feature(
                            bytes_list=tf.train.BytesList(
                                value=[name.encode("utf-8")]
                            )
                        ),
                        "label": tf.train.Feature(
                            int64_list=tf.train.Int64List(
                                value=label.tolist()
                            )
                        ),
                        "top": tf.train.Feature(
                            int64_list=tf.train.Int64List(
                                value=[top.min()]
                            )
                        ),
                        "left": tf.train.Feature(
                            int64_list=tf.train.Int64List(
                                value=[left.min()]
                            )
                        ),
                        "bottom": tf.train.Feature(
                            int64_list=tf.train.Int64List(
                                value=[bottom.max()]
                            )
                        ),
                        "right": tf.train.Feature(
                            int64_list=tf.train.Int64List(
                                value=[right.max()]
                            )
                        ),
                    }
                )
            ).SerializeToString()
        )
