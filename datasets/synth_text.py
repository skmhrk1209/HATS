import tensorflow as tf
import numpy as np
import sys
import os
import random
import functools
import itertools
import scipy.io
from algorithms import *


class Dataset(object):

    def __init__(self, filenames, num_epochs, batch_size, buffer_size,
                 image_size, channels_first, sequence_lengths):

        self.dataset = tf.data.TFRecordDataset(filenames)
        self.dataset = self.dataset.shuffle(
            buffer_size=buffer_size,
            reshuffle_each_iteration=True
        )
        self.dataset = self.dataset.repeat(num_epochs)
        self.dataset = self.dataset.map(
            map_func=functools.partial(
                self.parse,
                image_size=image_size,
                channels_first=channels_first,
                sequence_lengths=sequence_lengths
            ),
            num_parallel_calls=os.cpu_count()
        )
        self.dataset = self.dataset.batch(batch_size)
        self.dataset = self.dataset.prefetch(1)
        self.iterator = self.dataset.make_one_shot_iterator()

    def parse(self, example, image_size, channels_first, sequence_lengths):

        features = tf.parse_single_example(
            serialized=example,
            features={
                "path": tf.FixedLenFeature(
                    shape=[],
                    dtype=tf.string
                ),
                "label": tf.FixedLenFeature(
                    shape=[np.prod(sequence_lengths)],
                    dtype=tf.int64
                )
            }
        )

        image = tf.read_file(features["path"])
        image = tf.image.decode_jpeg(image, 3)
        image = tf.image.convert_image_dtype(image, tf.float32)

        if image_size:
            image = tf.image.resize_images(image, image_size)

        if channels_first:
            image = tf.transpose(image, [2, 0, 1])

        label = tf.cast(features["label"], tf.int32)
        label = tf.reshape(label, sequence_lengths)

        return {"image": image}, label

    def get_next(self):

        return self.iterator.get_next()


def convert_dataset(input_directory, output_filename, sequence_lengths):

    dataset = scipy.io.loadmat(os.path.join(input_directory, "gt.mat"))
    dataset = list(zip(dataset["imnames"][0],  dataset["txt"][0], dataset["wordBB"][0]))

    random.seed(0)
    random.shuffle(dataset)

    train_dataset = dataset[:int(len(dataset) * 0.9)]
    test_dataset = dataset[int(len(dataset) * 0.9):]

    root, ext = os.path.splitext(output_filename)
    train_filename = "{}_train{}".format(root, ext)
    test_filename = "{}_test{}".format(root, ext)

    class_ids = {}
    class_ids.update({chr(j): i for i, j in enumerate(range(ord(" "), ord("~") + 1), 0)})
    class_ids.update({"": max(class_ids.values()) + 1})

    def convert_dataset_impl(input_dataset, output_filename):

        with tf.python_io.TFRecordWriter(output_filename) as writer:

            for filenames, texts, bounding_boxes in input_dataset:

                bounding_box_indices = [0] + list(itertools.accumulate([len(text.split()) for text in texts]))[:-1]

                if len(bounding_boxes.shape) < 3:
                    bounding_boxes = bounding_boxes[:, :, np.newaxis]

                bounding_boxes = bounding_boxes.transpose([2, 1, 0])
                bounding_boxes = [bounding_boxes[i] for i in bounding_box_indices]

                texts = [text for text, bounding_box in sorted(zip(texts, bounding_boxes), key=lambda t: t[1][0][::-1].tolist())]
                texts = [[[char for char in string.strip(" ")] for string in sequence.split("\n")] for sequence in texts]

                label = map_innermost_element(lambda char: class_ids[char], texts)

                try:

                    for i, sequence_length in enumerate(sequence_lengths):

                        label = map_innermost_list(
                            function=lambda sequence: np.pad(
                                array=sequence,
                                pad_width=[[0, sequence_length - len(sequence)]] + [[0, 0]] * i,
                                mode="constant",
                                constant_values=class_ids[""]
                            ),
                            sequence=label
                        )

                except ValueError:

                    continue

                writer.write(
                    record=tf.train.Example(
                        features=tf.train.Features(
                            feature={
                                "path": tf.train.Feature(
                                    bytes_list=tf.train.BytesList(
                                        value=[os.path.join(input_directory, filenames[0]).encode("utf-8")]
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

    convert_dataset_impl(train_dataset, train_filename)
    convert_dataset_impl(test_dataset, test_filename)
