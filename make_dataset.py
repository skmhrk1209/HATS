from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import argparse
import os
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

    max_len = 5

    for struct in digit_struct.get_all_structs():

        if len(struct["label"]) > max_len:

            continue

        def random_resize_with_pad(image, size, mode="constant"):

            diff_y = size[0] - image.shape[0]
            diff_x = size[1] - image.shape[1]

            pad_width_y = np.random.randint(low=0, high=diff_y)
            pad_width_x = np.random.randint(low=0, high=diff_x)

            return np.pad(image, [[pad_width_y, diff_y - pad_width_y], [pad_width_x, diff_x - pad_width_x], [0, 0]], mode)

        def non_negative(x):

            return x if x > 0 else 0

        top = int(non_negative(min([_top for _top in struct["top"]])))
        bottom = int(non_negative(max([_top + _height for _top, _height in zip(struct["top"], struct["height"])])))

        left = int(non_negative(min([_left for _left in struct["left"]])))
        right = int(non_negative(max([_left + _width for _left, _width in zip(struct["left"], struct["width"])])))

        image = cv2.imread(os.path.join(os.path.dirname(args.struct), struct["name"]))
        image = cv2.resize(image[top:bottom, left:right, :], (28, 28))
        image = random_resize_with_pad(image, size=[128, 128], mode="edge")

        writer.write(
            record=tf.train.Example(
                features=tf.train.Features(
                    feature={
                        "image": tf.train.Feature(
                            bytes_list=tf.train.BytesList(
                                value=[image.tobytes()]
                            )
                        ),
                        "label": tf.train.Feature(
                            int64_list=tf.train.Int64List(
                                value=np.pad(
                                    array=map(int, struct["label"]),
                                    pad_width=[0, max_len - len(struct["label"])],
                                    mode="constant"
                                ).tolist()
                            )
                        )
                    }
                )
            ).SerializeToString()
        )
