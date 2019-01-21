import tensorflow as tf
import numpy as np
import glob
import sys
import os
from algorithms import *


def main(input_directory, output_filename, sequence_lengths):

    with tf.python_io.TFRecordWriter(output_filename) as writer:

        class_ids = {}
        class_ids.update({chr(j): i for i, j in enumerate(range(ord("0"), ord("9") + 1), 0)})
        class_ids.update({chr(j): i for i, j in enumerate(range(ord("A"), ord("Z") + 1), class_ids["9"] + 1)})
        class_ids.update({chr(j): i for i, j in enumerate(range(ord("a"), ord("z") + 1), class_ids["Z"] + 1)}),
        class_ids.update({"": max(class_ids.values()) + 1})

        input_filenames = glob.glob(os.path.join(input_directory, "*"))

        for input_filename in input_filenames:

            label = os.path.splitext(os.path.basename(input_filename))[0].split("_")[1:]
            label = map_innermost_element(list, label)
            label = map_innermost_element(lambda char: class_ids[char], label)

            for i, sequence_length in enumerate(sequence_lengths[::-1]):

                label = map_innermost_list(
                    function=lambda sequence: np.pad(
                        array=sequence,
                        pad_width=[[0, max(0, sequence_length - len(sequence))]] + [[0, 0]] * i,
                        mode="constant",
                        constant_values=class_ids[""]
                    ),
                    sequence=label
                )

            writer.write(
                record=tf.train.Example(
                    features=tf.train.Features(
                        feature={
                            "path": tf.train.Feature(
                                bytes_list=tf.train.BytesList(
                                    value=[input_filename.encode("utf-8")]
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
