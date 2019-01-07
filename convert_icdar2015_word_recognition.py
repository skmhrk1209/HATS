import tensorflow as tf
import numpy as np
import glob
import sys
import os
import re
from algorithms import *


def main(input_filename, output_filename, sequence_length):

    with tf.python_io.TFRecordWriter(output_filename) as writer:

        with open(input_filename) as input_file:

            regex = re.compile(r'(.+), "(.+)"')

            for line in input_file:

                filename, label = regex.findall(line.strip())[0]
                label = label.strip().strip('"')
                label = [ord(c) - 33 for c in label]
                label = np.pad(
                    array=label,
                    pad_width=[[0, sequence_length - len(label)]],
                    mode="constant",
                    constant_values=90
                )

                writer.write(
                    record=tf.train.Example(
                        features=tf.train.Features(
                            feature={
                                "path": tf.train.Feature(
                                    bytes_list=tf.train.BytesList(
                                        value=[filename.encode("utf-8")]
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


if __name__ == "__main__":

    main(*sys.argv[1:3], int(sys.argv[3]))
