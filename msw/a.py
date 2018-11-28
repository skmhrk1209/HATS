import tensorflow as tf
import numpy as np
import sys
import argparse
import itertools
import cv2
import glob
import functools
from attrdict import AttrDict
from dataset import Dataset
from model import Model
from networks.residual_network import ResidualNetwork
from networks.attention_network import AttentionNetwork

parser = argparse.ArgumentParser()
parser.add_argument("--model_dir", type=str, default="model", help="model directory")
parser.add_argument('--filenames', type=str, nargs="+", default=["a.tfrecord"], help="tfrecord filenames")
parser.add_argument("--num_epochs", type=int, default=10, help="number of training epochs")
parser.add_argument("--batch_size", type=int, default=128, help="batch size")
parser.add_argument("--buffer_size", type=int, default=900000, help="buffer size to shuffle dataset")
parser.add_argument("--train", action="store_true", help="with training")
parser.add_argument("--eval", action="store_true", help="with evaluation")
parser.add_argument("--predict", action="store_true", help="with prediction")
parser.add_argument("--gpu", type=str, default="0,1,2", help="gpu id")
args = parser.parse_args()

tf.logging.set_verbosity(tf.logging.INFO)

sys.setrecursionlimit(10000)


def search_bounding_box(image, threshold):

    if len(image.shape) == 3 and image.shape[-1] == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    max_value = 1.0 if np.issubdtype(image.dtype, np.floating) else 255
    binary = cv2.threshold(image, threshold, max_value, cv2.THRESH_BINARY)[1]
    flags = np.ones_like(binary, dtype=np.bool)
    h, w = binary.shape[:2]
    segments = []

    def depth_first_search(y, x):

        segments[-1].append((y, x))
        flags[y][x] = False

        for dy, dx in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
            if 0 <= y + dy < h and 0 <= x + dx < w:
                if flags[y + dy, x + dx] and binary[y + dy, x + dx]:
                    depth_first_search(y + dy, x + dx)

    for y in range(flags.shape[0]):
        for x in range(flags.shape[1]):
            if flags[y, x] and binary[y, x]:
                segments.append([])
                depth_first_search(y, x)

    bounding_boxes = [(lambda ls_1, ls_2: ((min(ls_1), min(ls_2)), (max(ls_1), max(ls_2))))(*zip(*segment)) for segment in segments]
    bounding_boxes = sorted(bounding_boxes, key=lambda box: abs(box[0][0] - box[1][0]) * abs(box[0][1] - box[1][1]))

    return bounding_boxes[-1]


class Dataset(object):

    def __init__(self, filenames, num_epochs, batch_size, buffer_size,
                 image_size, data_format, sequence_length, string_length):

        with tf.python_io.TFRecordWriter("a.tfrecord") as writer:

            for file in glob.glob("/home/sakuma/data/svt/*"):

                writer.write(
                    record=tf.train.Example(
                        features=tf.train.Features(
                            feature={
                                "path": tf.train.Feature(
                                    bytes_list=tf.train.BytesList(
                                        value=[file.encode("utf-8")]
                                    )
                                )
                            }
                        )
                    ).SerializeToString()
                )


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
            num_parallel_calls=os.cpu_count()
        )
        self.dataset = self.dataset.batch(batch_size)
        self.dataset = self.dataset.prefetch(1)
        self.iterator = self.dataset.make_one_shot_iterator()

    def parse(self, example, image_size, data_format, sequence_length, string_length):

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

        if data_format == "channels_first":
            image = tf.transpose(image, [2, 0, 1])

        label = tf.cast(features["label"], tf.int32)
        label = tf.reshape(label, [sequence_length, string_length])

        return {"image": image}, label

    def get_next(self):

        return self.iterator.get_next()


def main(unused_argv):

    classifier = tf.estimator.Estimator(
        model_fn=Model(
            convolutional_network=ResidualNetwork(
                conv_param=AttrDict(filters=64, kernel_size=[7, 7], strides=[2, 2]),
                pool_param=None,
                residual_params=[
                    AttrDict(filters=64, strides=[2, 2], blocks=2),
                    AttrDict(filters=128, strides=[2, 2], blocks=2),
                ],
                num_classes=None,
                data_format="channels_last"
            ),
            attention_network=AttentionNetwork(
                conv_params=[
                    AttrDict(filters=4, kernel_size=[9, 9], strides=[2, 2]),
                    AttrDict(filters=4, kernel_size=[9, 9], strides=[2, 2]),
                ],
                deconv_params=[
                    AttrDict(filters=16, kernel_size=[3, 3], strides=[2, 2]),
                    AttrDict(filters=16, kernel_size=[3, 3], strides=[2, 2]),
                ],
                rnn_params=[
                    AttrDict(sequence_length=4, num_units=[256]),
                    AttrDict(sequence_length=10, num_units=[256])
                ],
                data_format="channels_last"
            ),
            num_classes=63,
            num_tiles=1,
            data_format="channels_last",
            accuracy_type=Model.AccuracyType.FULL_SEQUENCE,
            hyper_params=AttrDict(
                cross_entropy_decay=1.0,
                attention_map_decay=0.001
            )
        ),
        model_dir=args.model_dir,
        config=tf.estimator.RunConfig().replace(
            session_config=tf.ConfigProto(
                gpu_options=tf.GPUOptions(
                    visible_device_list=args.gpu,
                    allow_growth=True
                )
            )
        )
    )

    if args.train:

        classifier.train(
            input_fn=lambda: Dataset(
                filenames=args.filenames,
                num_epochs=args.num_epochs,
                batch_size=args.batch_size,
                buffer_size=args.buffer_size,
                image_size=None,
                data_format="channels_last",
                sequence_length=4,
                string_length=10
            ).get_next()
        )

    if args.eval:

        eval_results = classifier.evaluate(
            input_fn=lambda: Dataset(
                filenames=args.filenames,
                num_epochs=args.num_epochs,
                batch_size=args.batch_size,
                buffer_size=args.buffer_size,
                image_size=None,
                data_format="channels_last",
                sequence_length=4,
                string_length=10
            ).get_next()
        )

        print(eval_results)

    if args.predict:

        predict_results = classifier.predict(
            input_fn=lambda: Dataset(
                filenames=args.filenames,
                num_epochs=args.num_epochs,
                batch_size=args.batch_size,
                buffer_size=args.buffer_size,
                image_size=[256, 256],
                data_format="channels_last",
                sequence_length=4,
                string_length=10
            ).get_next()
        )

        def scale(input, input_min, input_max, output_min, output_max):
            return output_min + (input - input_min) / (input_max - input_min) * (output_max - output_min)

        for i, predict_result in enumerate(itertools.islice(predict_results, 10)):

            for j in range(1):

                attention_map_images = []
                boundin_box_images = []

                for k in range(4):

                    attention_map_images.append([])
                    boundin_box_images.append([])

                    for l in range(10):

                        merged_attention_map = predict_result["merged_attention_maps"][j, k, l]
                        merged_attention_map = scale(merged_attention_map, merged_attention_map.min(), merged_attention_map.max(), 0.0, 1.0)
                        merged_attention_map = cv2.resize(merged_attention_map, (256, 256))
                        bounding_box = search_bounding_box(merged_attention_map, 0.5)

                        attention_map_image = np.copy(predict_result["images"][j])
                        attention_map_image += np.pad(np.expand_dims(merged_attention_map, axis=-1), [[0, 0], [0, 0], [0, 2]], "constant")
                        attention_map_images[-1].append(attention_map_image)

                        boundin_box_image = np.copy(predict_result["images"][j])
                        boundin_box_image = cv2.rectangle(boundin_box_image, bounding_box[0][::-1], bounding_box[1][::-1], (255, 0, 0), 2)
                        boundin_box_images[-1].append(boundin_box_image)

                attention_map_images = np.concatenate([
                    np.concatenate(attention_map_images, axis=1)
                    for attention_map_images in attention_map_images
                ], axis=0)

                boundin_box_images = np.concatenate([
                    np.concatenate(boundin_box_images, axis=1)
                    for boundin_box_images in boundin_box_images
                ], axis=0)

                attention_map_images = cv2.cvtColor(attention_map_images, cv2.COLOR_BGR2RGB)
                boundin_box_images = cv2.cvtColor(boundin_box_images, cv2.COLOR_BGR2RGB)

                attention_map_images = scale(attention_map_images, 0.0, 1.0, 0.0, 255.0)
                boundin_box_images = scale(boundin_box_images, 0.0, 1.0, 0.0, 255.0)

                cv2.imwrite("a/attention_map_{}_{}.jpg".format(i, j), attention_map_images)
                cv2.imwrite("a/boundin_box_{}_{}.jpg".format(i, j), boundin_box_images)


if __name__ == "__main__":
    tf.app.run()
