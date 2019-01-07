import tensorflow as tf
import numpy as np
import glob
import sys
import os
from algorithms import *


def main(input_filename, output_filename, sequence_length):

    with tf.python_io.TFRecordWriter(output_filename) as writer:

        with open(input_filename) as input_file:

            for line in input_file:

                filename, label = line.split()
                filename = filename.strip(",")
                label = label.strip('"')
                label = [ord(c) - 33 for c in label]
                label = map(
                    function=lambda sequence: np.pad(
                        array=sequence,
                        pad_width=[[0, sequence_length - len(sequence)]],
                        mode="constant",
                        constant_values=90
                    ),
                    sequence=label
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

    main(*sys.argv[1:3], list(map(int, sys.argv[3:])))
