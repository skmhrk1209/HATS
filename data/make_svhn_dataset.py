import tensorflow as tf
import numpy as np
import argparse
import os
import glob
import h5py

parser = argparse.ArgumentParser()
parser.add_argument("--filename", type=str, required=True, help="tfrecord filename")
parser.add_argument("--directory", type=str, required=True, help="path to data directory")
args = parser.parse_args()


class DigitStruct:

    def __init__(self, file):

        self.file = h5py.File(file, "r")
        self.digit_struct_name = self.file["digitStruct"]["name"]
        self.digit_struct_bbox = self.file["digitStruct"]["bbox"]

    def get_name(self, index):

        return "".join([chr(c[0]) for c in self.file[self.digit_struct_name[index][0]].value])

    def get_attr(self, attr):

        return [self.file[attr.value[i].item()].value[0][0]
                for i in range(len(attr))] if len(attr) > 1 else [attr.value[0][0]]

    def get_bbox(self, index):

        return {attr: self.get_attr(self.file[self.digit_struct_bbox[index].item()][attr])
                for attr in ["label", "top", "left", "height", "width"]}

    def get_digit_struct(self, index):

        def concat(dicts):
            concated = {}
            for dict in dicts:
                concated.update(dict)
            return concated

        return concat([self.get_bbox(index), {"name": self.get_name(index)}])

    def get_all_digit_structs(self):

        return [self.get_digit_struct(i) for i in range(len(self.digit_struct_name))]


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
