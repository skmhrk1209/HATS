import tensorflow as tf
import numpy as np
import functools


class Dataset(object):

    def __init__(self, filenames, num_epochs, batch_size, buffer_size, num_cpus,
                 image_size, data_format, sequence_length=9, string_length=35):

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
                sequence_length=sequence_length,
                string_length=string_length
            ),
            num_parallel_calls=num_cpus
        )
        self.dataset = self.dataset.batch(batch_size)
        self.dataset = self.dataset.prefetch(1)
        self.iterator = self.dataset.make_one_shot_iterator()

    def parse(self, example, image_size, data_format, sequence_length, string_length):

        features = tf.parse_single_example(
            serialized=example,
            features={
                "image/encoded": tf.FixedLenFeature(
                    shape=[],
                    dtype=tf.string
                ),
                "image/class": tf.FixedLenFeature(
                    shape=[37],
                    dtype=tf.int64
                )
            }
        )

        image = tf.image.decode_png(features["image/encoded"], 3)
        image = tf.image.convert_image_dtype(image, tf.float32)
        image.set_shape([150, 600, 3])

        if image_size:
            image = tf.image.resize_images(image, image_size)

        if data_format == "channels_first":
            image = tf.transpose(image, [2, 0, 1])

        label = tf.cast(features["image/class"], tf.int32)
        indices = tf.cast(tf.squeeze(tf.where(tf.not_equal(label, 133))), tf.int32)
        label = tf.gather(label, indices)
        label = tf.concat([[0], label, [0]], axis=0)
        indices = tf.cast(tf.squeeze(tf.where(tf.equal(label, 0))), tf.int32)
        indices = tf.stack([indices[:-1] + 1, indices[1:]], axis=-1)

        label = tf.map_fn(
            fn=lambda range: (lambda label: tf.pad(
                tensor=label,
                paddings=[[0, string_length - tf.shape(label)[0]]],
                mode="constant",
                constant_values=133
            ))(label[range[0]: range[1]]),
            elems=indices
        )

        label = (lambda label: tf.pad(
            tensor=label,
            paddings=[[0, sequence_length - tf.shape(label)[0]], [0, 0]],
            mode="constant",
            constant_values=133
        ))(label)

        return {"image": image}, label

    def get_next(self):

        return self.iterator.get_next()
