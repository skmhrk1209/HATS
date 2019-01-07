import tensorflow as tf
import numpy as np
import glob
import sys
import os
import re
from algorithms import *


def main(input_directory, output_filename, sequence_length):

    with tf.python_io.TFRecordWriter(output_filename) as writer:

        with open(os.path.join(input_directory, "gt.txt")) as f:

            regex = re.compile(r'(.+), "(.+)"')

            for line in f:

                filename, label = regex.findall(line.strip())[0]
                label = label.strip().strip('"')
                label = [ord(c) - 32 for c in label]
                if any([i > 95 for i in label]):
                    continue
                label = np.pad(
                    array=label,
                    pad_width=[[0, sequence_length - len(label)]],
                    mode="constant",
                    constant_values=95
                )

                writer.write(
                    record=tf.train.Example(
                        features=tf.train.Features(
                            feature={
                                "path": tf.train.Feature(
                                    bytes_list=tf.train.BytesList(
                                        value=[os.path.join(input_directory, filename).encode("utf-8")]
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
