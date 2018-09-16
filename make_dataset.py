from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
import functools
import itertools
import operator
import argparse
import glob
import os
import sys
import cv2
import h5py

parser = argparse.ArgumentParser()
parser.add_argument("struct", type=str)
parser.add_argument("filename", type=str)
args = parser.parse_args()


class DigitStruct:

    def __init__(self, file):

        self.file = h5py.File(file, "r")
        self.name = self.file["digitStruct"]["name"]
        self.bbox = self.file["digitStruct"]["bbox"]

    def get_name(self, index):

        return "".join([chr(c[0]) for c in self.file[self.name[index][0]].value])

    def get_values(self, attr):

        return [self.file[attr.value[j].item()].value[0][0] for j in range(len(attr))] if len(attr) > 1 else [attr.value[0][0]]

    def get_bbox(self, index):

        return {key: self.get_values(self.file[self.bbox[index].item()][key]) for key in ["label", "top", "left", "height", "width"]}

    def get_struct(self, index):

        def combine(dicts):

            combined = {}

            for dict in dicts:

                combined.update(dict)

            return combined

        return combine([self.get_bbox(index), {"name": self.get_name(index)}])

    def get_all_structs(self):

        return [self.get_struct(i) for i in range(len(self.name))]


with tf.python_io.TFRecordWriter(args.filename) as writer:

    digit_struct = DigitStruct(args.struct)

    for struct in digit_struct.get_all_structs():

        def pad(list, length, value):

            while len(list) < length:

                list.append(value)

            return list

        writer.write(
            record=tf.train.Example(
                features=tf.train.Features(
                    feature={
                        "path": tf.train.Feature(
                            bytes_list=tf.train.BytesList(
                                value=[os.path.join(os.path.dirname(args.struct), struct["name"]).encode("utf-8")]
                            )
                        ),
                        "length": tf.train.Feature(
                            int64_list=tf.train.Int64List(
                                value=[len(struct["label"])]
                            )
                        ),
                        "label": tf.train.Feature(
                            int64_list=tf.train.Int64List(
                                value=pad(map(lambda label: int(label) % 10, struct["label"]), 5, 10)
                            )
                        ),
                        "top": tf.train.Feature(
                            int64_list=tf.train.Int64List(
                                value=[int(min(struct["top"]))]
                            )
                        ),
                        "bottom": tf.train.Feature(
                            int64_list=tf.train.Int64List(
                                value=[int(max([top + height for top, height in zip(struct["top"], struct["height"])]))]
                            )
                        ),
                        "left": tf.train.Feature(
                            int64_list=tf.train.Int64List(
                                value=[int(min(struct["left"]))]
                            )
                        ),
                        "right": tf.train.Feature(
                            int64_list=tf.train.Int64List(
                                value=[int(max([left + width for left, width in zip(struct["left"], struct["width"])]))]
                            )
                        )
                    }
                )
            ).SerializeToString()
        )
