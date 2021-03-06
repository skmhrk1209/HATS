import tensorflow as tf
import numpy as np
import skimage
import argparse
import sys
import os
from tqdm import *
from algorithms import *

parser = argparse.ArgumentParser()
parser.add_argument("--input_filename", type=str, help="input ground truth filename")
parser.add_argument("--output_filename", type=str, help="output tfrecord filename")
parser.add_argument("--num_words", type=int, help="number of words contained in a instance (include eos)")
parser.add_argument("--num_chars", type=int, help="number of characters contained in a instance (include eos)")
args = parser.parse_args()


def pad(sequence, sequence_length, value):
    while len(sequence) < sequence_length:
        sequence.append(value)
    return sequence


def invalid(path):
    try:
        skimage.io.imread(path)
    except:
        return True
    return False


def main(input_filename, output_filename, num_words, num_chars):

    class_ids = {}
    class_ids.update({chr(j): i for i, j in enumerate(range(ord("0"), ord("9") + 1), 0)})
    class_ids.update({chr(j): i for i, j in enumerate(range(ord("A"), ord("Z") + 1), class_ids["9"] + 1)})
    class_ids.update({"": max(class_ids.values()) + 1})

    with tf.python_io.TFRecordWriter(output_filename) as writer:

        with open(input_filename) as f:

            for line in tqdm(f):

                path, words = line.split()
                path = os.path.join(os.path.dirname(input_filename), path)

                if invalid(path):
                    print("invalid file: {}".format(path))
                    continue

                words = words.split("_")
                words = map_innermost_list(lambda words: pad(words, num_words, ""), words)
                words = map_innermost_element(lambda word: word.upper(), words)
                chars = map_innermost_element(lambda word: list(word), words)
                chars = map_innermost_list(lambda chars: pad(chars, num_chars, ""), chars)
                label = map_innermost_element(lambda char: class_ids[char], chars)
                label = flatten_innermost_element(label)

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
                                        value=label
                                    )
                                )
                            }
                        )
                    ).SerializeToString()
                )


if __name__ == "__main__":

    main(args.input_filename, args.output_filename, args.num_words, args.num_chars)
