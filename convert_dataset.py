import tensorflow as tf
import numpy as np
import glob
import sys
import os
from tqdm import *
from algorithms import *


def main(input_filename, output_filename, num_words, num_chars):

    class_ids = {}
    class_ids.update({chr(j): i for i, j in enumerate(range(ord("0"), ord("9") + 1), 0)})
    class_ids.update({chr(j): i for i, j in enumerate(range(ord("A"), ord("Z") + 1), class_ids["9"] + 1)})
    class_ids.update({"!": max(class_ids.values()) + 1})
    class_ids.update({"": max(class_ids.values()) + 1})

    with tf.python_io.TFRecordWriter(output_filename) as writer:

        with open(input_filename) as f:

            for line in tqdm(f):

                path, words = line.split()
                path = os.path.join(os.path.dirname(sys.argv[1]), path)
                words = words.split("_")
                words = map_innermost_list(lambda words: np.pad(
                    words, [[0, num_words - len(words)]],
                    "constant", constant_values=""
                ), words)
                words = map_innermost_element(lambda word: word + "!", words)
                words = map_innermost_element(lambda word: word.upper(), words)
                chars = map_innermost_element(lambda word: list(word), words)
                chars = map_innermost_list(lambda chars: np.pad(
                    chars, [[0, num_chars - len(chars)]],
                    "constant", constant_values=""
                ), chars)
                chars = map_innermost_element(lambda char: class_ids[char], chars)

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

    print(sys.argv)

    main(*sys.argv[1:3], list(map(int, sys.argv[3:])))
