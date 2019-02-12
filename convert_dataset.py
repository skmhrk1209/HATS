import tensorflow as tf
import numpy as np
import glob
import sys
import os
from tqdm import *
from algorithms import *


def main(input_filename, output_filename, seq_lens):

    class_ids = {}
    class_ids.update({chr(j): i for i, j in enumerate(range(ord("0"), ord("9") + 1), 0)})
    class_ids.update({chr(j): i for i, j in enumerate(range(ord("A"), ord("Z") + 1), class_ids["9"] + 1)})
    class_ids.update({"": max(class_ids.values()) + 1})

    with tf.python_io.TFRecordWriter(output_filename) as writer:

        with open(input_filename) as f:

            for line in tqdm(f):

                path, label = line.split()
                path = os.path.join(os.path.dirname(sys.argv[1]), path)
                label = label.split("_")
                label = map_innermost_element(lambda string: string.upper(), label)
                label = map_innermost_element(lambda string: list(string), label)

                try:
                    label = map_innermost_element(lambda char: class_ids[char], label)
                except KeyError as error:
                    print("{} at {}".format(error, path))
                    continue

                for i, seq_len in enumerate(seq_lens[::-1]):

                    label = map_innermost_list(
                        func=lambda seq: np.pad(
                            array=seq,
                            pad_width=[[0, seq_len - len(seq)]] + [[0, 0]] * i,
                            mode="constant",
                            constant_values=class_ids[""]
                        ),
                        seq=label
                    )

                writer.write(
                    record=tf.train.Example(
                        features=tf.train.Features(
                            feature={
                                "path": tf.train.Feature(
                                    bytes_list=tf.train.BytesList(
                                        value=[path.encode("utf-8")]
                                    )
                                ),
                                "label": tf.train.Feature(
                                    int64_list=tf.train.Int64List(
                                        value=label.reshape([-1]).tolist()
                                    )
                                )
                            }
                        )
                    ).SerializeToString()
                )


if __name__ == "__main__":

    main(*sys.argv[1:3], list(map(int, sys.argv[3:])))
