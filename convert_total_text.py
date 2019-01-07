import tensorflow as tf
import numpy as np
import scipy.io
import glob
import sys
import os
from algorithms import *


def main(input_directory, output_filename, sequence_lengths):

    with tf.python_io.TFRecordWriter(output_filename) as writer:

        input_filenames = glob.glob(os.path.join(input_directory, "*"))
        input_datasets = [scipy.io.loadmat(input_filename) for input_filename in input_filenames]

        for input_filename, input_dataset in zip(input_filenames, input_datasets):

            label = list(map("".join, list(zip(*sorted(zip(*input_dataset["rectgt"][:, [1, 0, 6]].T))))[-1]))
            label = map_innermost_element(list, label)
            label = map_innermost_element(lambda c: ord(c) - 32, label)

            for i, sequence_length in enumerate(sequence_lengths[::-1]):

                label = map_innermost_list(
                    function=lambda sequence: np.pad(
                        array=sequence,
                        pad_width=[[0, sequence_length - len(sequence)]] + [[0, 0]] * i,
                        mode="constant",
                        constant_values=95
                    ) if len(sequence) < sequence_length else np.array(
                        object=sequence[:sequence_length]
                    ),
                    sequence=label
                )

            writer.write(
                record=tf.train.Example(
                    features=tf.train.Features(
                        feature={
                            "path": tf.train.Feature(
                                bytes_list=tf.train.BytesList(
                                    value=[input_filename.replace("labels", "images").replace("mat", "jpg").encode("utf-8")]
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


if __name__ == "__main__":

    main(*sys.argv[1:3], list(map(int, sys.argv[3:])))