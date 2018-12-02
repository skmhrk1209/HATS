import tensorflow as tf
import numpy as np
import functools
import glob
import os


class Dataset(object):

    def __init__(self, filenames, num_epochs, batch_size, buffer_size,
                 image_size, data_format, string_length):

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
                data_format=data_format,
                string_length=string_length
            ),
            num_parallel_calls=os.cpu_count()
        )
        self.dataset = self.dataset.batch(batch_size)
        self.dataset = self.dataset.prefetch(1)
        self.iterator = self.dataset.make_one_shot_iterator()

    def parse(self, example, image_size, data_format, string_length):

        features = tf.parse_single_example(
            serialized=example,
            features={
                "path": tf.FixedLenFeature(
                    shape=[],
                    dtype=tf.string
                ),
                "label": tf.FixedLenFeature(
                    shape=[string_length],
                    dtype=tf.int64
                )
            }
        )

        image = tf.read_file(features["path"])
        image = tf.image.decode_jpeg(image, 3)
        image = tf.image.convert_image_dtype(image, tf.float32)

        if image_size:
            image = tf.image.resize_images(image, image_size)

        if data_format == "channels_first":
            image = tf.transpose(image, [2, 0, 1])

        label = tf.cast(features["label"], tf.int32)

        return {"image": image}, label

    def get_next(self):

        return self.iterator.get_next()


def convert_dataset(input_directory, output_filename, sequence_length, string_length):

    with tf.python_io.TFRecordWriter(output_filename) as writer:

        class_ids = {}

        for i in range(ord("0"), ord("z") + 1):

            if ord("0") <= i <= ord("9"):
                class_ids[chr(i)] = i - ord("0")
            elif ord("A") <= i <= ord("Z"):
                class_ids[chr(i)] = i - ord("A") + class_ids["9"] + 1
            elif ord("a") <= i <= ord("z"):
                class_ids[chr(i)] = i - ord("a") + class_ids["Z"] + 1

        class_ids[""] = max(class_ids.values()) + 1

        for filename in glob.glob(os.path.join(input_directory, "*")):

            string = os.path.splitext(os.path.basename(filename))[0].split("_")[1]

            label = np.pad(
                array=[class_ids[char] for char in string],
                pad_width=[[0, string_length - len(string)]],
                mode="constant",
                constant_values=class_ids[""]
            )

            writer.write(
                record=tf.train.Example(
                    features=tf.train.Features(
                        feature={
                            "path": tf.train.Feature(
                                bytes_list=tf.train.BytesList(
                                    value=[file.encode("utf-8")]
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
