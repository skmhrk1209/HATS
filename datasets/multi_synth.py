import tensorflow as tf
import numpy as np
import functools
import glob
import os
import random
import threading
import cv2
from numba import jit
from tqdm import tqdm, trange
from shapely.geometry import box


class Dataset(object):

    def __init__(self, filenames, num_epochs, batch_size, buffer_size,
                 image_size, channels_first, sequence_length, string_length):

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
                sequence_length=sequence_length,
                string_length=string_length
            ),
            num_parallel_calls=os.cpu_count()
        )
        self.dataset = self.dataset.batch(batch_size)
        self.dataset = self.dataset.prefetch(1)
        self.iterator = self.dataset.make_one_shot_iterator()

    def parse(self, example, image_size, channels_first, sequence_length, string_length):

        features = tf.parse_single_example(
            serialized=example,
            features={
                "path": tf.FixedLenFeature(
                    shape=[],
                    dtype=tf.string
                ),
                "label": tf.FixedLenFeature(
                    shape=[sequence_length * string_length],
                    dtype=tf.int64
                )
            }
        )

        image = tf.read_file(features["path"])
        image = tf.image.decode_jpeg(image, 3)
        image = tf.image.convert_image_dtype(image, tf.float32)
        image.set_shape([256, 256, 3])

        if image_size:
            image = tf.image.resize_images(image, image_size)

        if channels_first:
            image = tf.transpose(image, [2, 0, 1])

        label = tf.cast(features["label"], tf.int32)
        label = tf.reshape(label, [sequence_length, string_length])

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

        for input_filename in glob.glob(os.path.join(input_directory, "*")):

            strings = os.path.splitext(os.path.basename(input_filename))[0].split("_")[1:]

            label = np.pad(
                array=[
                    np.pad(
                        array=[class_ids[char] for char in string],
                        pad_width=[[0, string_length - len(string)]],
                        mode="constant",
                        constant_values=class_ids[""]
                    ) for string in strings
                ],
                pad_width=[[0, sequence_length - len(strings)], [0, 0]],
                mode="constant",
                constant_values=class_ids[""]
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


def make_dataset(input_directory, output_directory, num_data, image_size, sequence_length, string_length, num_retries):

    input_filenames = [
        filename for filename in tqdm(glob.glob(os.path.join(input_directory, "*")))
        if ((lambda string: len(string) <= string_length)(os.path.splitext(os.path.basename(filename))[0].split("_")[1]) and
            (lambda image: image is not None and all([l1 <= l2 for l1, l2 in zip(image.shape[:2], image_size)]))(cv2.imread(filename)))
    ]

    random.seed(0)
    random.shuffle(input_filenames)

    multi_thread(make_dataset_impl, num_threads=os.cpu_count())(
        input_filenames=input_filenames,
        output_directory=output_directory,
        num_data=num_data,
        image_size=image_size,
        sequence_length=sequence_length,
        num_retries=num_retries
    )


@jit(nopython=False, nogil=True)
def make_dataset_impl(input_filenames, output_directory, num_data, image_size, sequence_length, num_retries, thread_id):

    for i in trange(num_data * thread_id, num_data * (thread_id + 1)):

        output_image = np.zeros(image_size + [3], dtype=np.uint8)

        strings = []
        rects = []

        for input_filename in random.sample(input_filenames, random.randint(1, sequence_length)):

            string = os.path.splitext(os.path.basename(input_filename))[0].split("_")[1]
            input_image = cv2.imread(input_filename)

            for _ in range(num_retries):

                h = input_image.shape[0]
                w = input_image.shape[1]
                y = random.randint(0, image_size[0] - h)
                x = random.randint(0, image_size[1] - w)
                proposal = (y, x, y + h, x + w)

                for rect in rects:
                    if box(*proposal).intersects(box(*rect)):
                        break

                else:
                    output_image[y:y+h, x:x+w, :] += input_image
                    strings.append(string)
                    rects.append(proposal)
                    break

        strings = [string for rect, string in sorted(zip(rects, strings))]
        output_filename = "{}_{}.jpg".format(i, "_".join(strings))
        cv2.imwrite(os.path.join(output_directory, output_filename), output_image)


def multi_thread(func, num_threads):

    def func_mt(*args, **kwargs):

        threads = [
            threading.Thread(
                target=func,
                args=args,
                kwargs=dict(kwargs, thread_id=i)
            ) for i in range(num_threads)
        ]

        for thread in threads:
            thread.start()

        for thread in threads:
            thread.join()

    return func_mt
