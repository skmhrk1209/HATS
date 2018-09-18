from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import argparse
import subprocess
import os
import cv2
import h5py

parser = argparse.ArgumentParser()
parser.add_argument("struct", type=str)
parser.add_argument("filename", type=str)
args = parser.parse_args()

dirname = os.path.dirname(os.path.abspath(__file__))


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

    max_length = 5

    for struct in digit_struct.get_all_structs():

        length = len(struct["label"])

        if length > max_length:

            continue

        def clip(x, min_value, max_value):

            return min(max_value, max(min_value, x))

        input = cv2.imread(os.path.join(os.path.dirname(args.struct), struct["name"]))

        top = int(max(min([t for t in struct["top"]]), 0))
        bottom = int(min(max([t + h for t, h in zip(struct["top"], struct["height"])]), input.shape[0]))
        left = int(max(min([l for l in struct["left"]]), 0))
        right = int(min(max([l + w for l, w in zip(struct["left"], struct["width"])]), input.shape[1]))

        mask = np.zeros_like(input)
        mask[top:bottom, left:right, :] = 255

        cv2.imwrite("input.png", input)
        cv2.imwrite("mask.png", mask)

        try:

            subprocess.call([
                "th", "inpaint.lua",
                "--input",  os.path.join(dirname, "input.png"),
                "--mask", os.path.join(dirname, "mask.png"),
                "--output", os.path.join(dirname, "output.png"),
                "--maxdim", str(max(input.shape[:2]))
            ], cwd="../siggraph2017_inpainting")

        except subprocess.CalledProcessError as error:

            print(error.output)

            continue

        output = cv2.imread("output.png")
        output = cv2.resize(output, (128, 128))

        bounding_box = input[top:bottom, left:right, :]
        bounding_box = cv2.resize(bounding_box, (28, 28))

        y = np.random.randint(0, 100)
        x = np.random.randint(0, 100)

        output[y:y+28, x:x+28, :] = bounding_box

        label = np.array(struct["label"]).astype(np.int32) % 10
        label = np.pad(
            array=label,
            pad_width=[0, max_length - length],
            mode="constant",
            constant_values=10
        )

        writer.write(
            record=tf.train.Example(
                features=tf.train.Features(
                    feature={
                        "image": tf.train.Feature(
                            bytes_list=tf.train.BytesList(
                                value=[output.tobytes()]
                            )
                        ),
                        "label": tf.train.Feature(
                            int64_list=tf.train.Int64List(
                                value=label.tolist()
                            )
                        )
                    }
                )
            ).SerializeToString()
        )
