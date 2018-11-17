import tensorflow as tf
import numpy as np
import argparse
import itertools
import cv2
from attrdict import AttrDict
from data.multi_mjsynth import Dataset
from models.acnn import ACNN
from networks.residual_network import ResidualNetwork
from networks.attention_network import AttentionNetwork

parser = argparse.ArgumentParser()
parser.add_argument("--model_dir", type=str, default="multi_mjsynth_acnn_model_1", help="model directory")
parser.add_argument('--filenames', type=str, nargs="+", default=["multi_mjsynth/train.tfrecord"], help="tfrecord filenames")
parser.add_argument("--num_epochs", type=int, default=10, help="number of training epochs")
parser.add_argument("--batch_size", type=int, default=128, help="batch size")
parser.add_argument("--buffer_size", type=int, default=900000, help="buffer size to shuffle dataset")
parser.add_argument("--num_cpus", type=int, default=32, help="number of logical processors")
parser.add_argument("--train", action="store_true", help="with training")
parser.add_argument("--eval", action="store_true", help="with evaluation")
parser.add_argument("--predict", action="store_true", help="with prediction")
parser.add_argument("--gpu", type=str, default="0,1,2", help="gpu id")
args = parser.parse_args()

tf.logging.set_verbosity(tf.logging.INFO)


def scale(input, input_min, input_max, output_min, output_max):

    return output_min + (input - input_min) / (input_max - input_min) * (output_max - output_min)


def search_bounding_box(image, method, threshold):

    assert(method in ["segment", "density"])

    if len(image.shape) == 3 and image.shape[-1] == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    if image.dtype == np.uint8:
        image = image / 255.
        threshold = threshold / 255.

    def segment_search_bounding_box(image, threshold):

        binary = cv2.threshold(image, threshold, 1.0, cv2.THRESH_BINARY)[1]
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

    def density_search_bounding_box(image, threshold):

        mu = cv2.moments(image, False)
        y, x = (int(mu["m01"] / mu["m00"]), int(mu["m10"] / mu["m00"]))
        h, w = image.shape[:2]
        dy, dx = (1, 1)

        while image[y - dy: y + dy, x - dx: x + dx].mean() > threshold:

            if dy >= min(y, h - y - 1) and dx >= min(x, w - x - 1):
                break

            elif dy >= min(y, h - y - 1):
                dx += 1

            elif dx >= min(x, w - x - 1):
                dy += 1

            else:
                density_1 = image[y - dy - 1: y + dy + 1, x - dx: x + dx].mean()
                density_2 = image[y - dy: y + dy, x - dx - 1: x + dx + 1].mean()

                if density_1 > density_2:
                    dy += 1
                else:
                    dx += 1

        return ((y - dy, x - dx), (y + dy, x + dx))

    return (segment_search_bounding_box if method == "segment" else density_search_bounding_box)(image, threshold)


def main(unused_argv):

    num_steps = args.buffer_size * args.num_epochs / args.batch_size

    # best model (accuracy: 86.6 %)
    multi_mjsynth_classifier = tf.estimator.Estimator(
        model_fn=ACNN(
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
                    # kernel size shuld be large
                    AttrDict(filters=4, kernel_size=[9, 9], strides=[2, 2]),
                    AttrDict(filters=4, kernel_size=[9, 9], strides=[2, 2]),
                ],
                deconv_params=[
                    AttrDict(filters=16, kernel_size=[3, 3], strides=[2, 2]),
                    AttrDict(filters=16, kernel_size=[3, 3], strides=[2, 2]),
                ],
                rnn_params=[
                    AttrDict(sequence_length=4, num_units=[256]),
                    AttrDict(sequence_length=10, num_units=[16])
                ],
                data_format="channels_last"
            ),
            num_classes=63,
            data_format="channels_last",
            hyper_params=AttrDict(
                cross_entropy_decay=lambda global_step: 1e-0,
                attention_map_decay=lambda global_step: 1e-3,
                total_variation_decay=lambda global_step: 1e-9
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

        multi_mjsynth_classifier.train(
            input_fn=lambda: Dataset(
                filenames=args.filenames,
                num_epochs=args.num_epochs,
                batch_size=args.batch_size,
                buffer_size=args.buffer_size,
                num_cpus=args.num_cpus,
                image_size=None,
                data_format="channels_last",
                sequence_length=4,
                string_length=10
            ).get_next(),
            hooks=[
                tf.train.LoggingTensorHook(
                    tensors={
                        "accuracy": "accuracy_value"
                    },
                    every_n_iter=100
                )
            ]
        )

    if args.eval:

        eval_results = multi_mjsynth_classifier.evaluate(
            input_fn=lambda: Dataset(
                filenames=args.filenames,
                num_epochs=args.num_epochs,
                batch_size=args.batch_size,
                buffer_size=args.buffer_size,
                num_cpus=args.num_cpus,
                image_size=None,
                data_format="channels_last",
                sequence_length=4,
                string_length=10
            ).get_next()
        )

        print(eval_results)

    if args.predict:

        predict_results = multi_mjsynth_classifier.predict(
            input_fn=lambda: Dataset(
                filenames=args.filenames,
                num_epochs=args.num_epochs,
                batch_size=args.batch_size,
                buffer_size=args.buffer_size,
                num_cpus=args.num_cpus,
                image_size=None,
                data_format="channels_last",
                sequence_length=4,
                string_length=10
            ).get_next()
        )

        for i, predict_result in enumerate(itertools.islice(predict_results, 10)):

            merged_attention_maps = [
                (name, tensor) for name, tensor in predict_result.items()
                if "merged_attention_maps" in name
            ]

            for name, merged_attention_map in merged_attention_maps:

                merged_attention_map = scale(merged_attention_map, merged_attention_map.min(), merged_attention_map.max(), 0.0, 1.0)
                merged_attention_map = cv2.resize(merged_attention_map, (256, 256))
                bounding_box = search_bounding_box(merged_attention_map, 0.5)

                image = predict_result["images"]
                image = cv2.rectangle(image, bounding_box[0], bounding_box[1], (255, 0, 0))
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

                cv2.imwrite(
                    "outputs/multi_mjsynth/merged_attention_map_{}.png".format("_".join(name.split("_")[1:])),
                    scale(merged_attention_map, 0.0, 1.0, 0.0, 255.0)
                )
                cv2.imwrite(
                    "outputs/multi_mjsynth/image_{}.png".format("_".join(name.split("_")[1:])),
                    scale(image, 0.0, 1.0, 0.0, 255.0)
                )


if __name__ == "__main__":
    tf.app.run()
