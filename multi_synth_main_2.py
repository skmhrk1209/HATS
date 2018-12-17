import tensorflow as tf
import numpy as np
import sys
import argparse
import itertools
import math
import cv2
import image as img
from attrdict import AttrDict
from dataset import Dataset
from model_ import Model
from networks.residual_network import ResidualNetwork
from networks.attention_network import AttentionNetwork

parser = argparse.ArgumentParser()
parser.add_argument("--model_dir", type=str, default="multi_synth_acnn_model_2", help="model directory")
parser.add_argument('--filenames', type=str, nargs="+", default=["multi_synth_train.tfrecord"], help="tfrecord filenames")
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
                channels_first=False
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
                    AttrDict(sequence_length=4, num_units=[256, 256]),
                    AttrDict(sequence_length=10, num_units=[256, 256])
                ],
                channels_first=False
            ),
            num_classes=63,
            channels_first=False,
            accuracy_type=Model.AccuracyType.EDIT_DISTANCE,
            hyper_params=AttrDict(attention_map_decay=0.001)
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
                image_size=[256, 256],
                channels_first=False,
                sequence_lengths=[4, 10]
            ).get_next()
        )

    if args.eval:

        eval_results = classifier.evaluate(
            input_fn=lambda: Dataset(
                filenames=args.filenames,
                num_epochs=args.num_epochs,
                batch_size=args.batch_size,
                buffer_size=args.buffer_size,
                image_size=[256, 256],
                channels_first=False,
                sequence_lengths=[4, 10]
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
                channels_first=False,
                sequence_lengths=[4, 10]
            ).get_next()
        )

        class_ids = {}
        class_ids.update({chr(j): i for i, j in enumerate(range(ord("0"), ord("9") + 1), 0)})
        class_ids.update({chr(j): i for i, j in enumerate(range(ord("A"), ord("Z") + 1), class_ids["9"] + 1)})
        class_ids.update({chr(j): i for i, j in enumerate(range(ord("a"), ord("z") + 1), class_ids["Z"] + 1)}),
        class_ids.update({"": max(class_ids.values()) + 1})

        class_chars = dict(map(lambda key_value: key_value[::-1], class_ids.items()))

        for predict_result in itertools.islice(predict_results, 10):

            attention_map_images = []
            bounding_box_images = []

            for i in range(predict_result["attention_maps"].shape[0]):

                attention_map_images.append([])
                bounding_box_images.append([])

                for j in range(predict_result["attention_maps"].shape[1]):

                    attention_map = predict_result["attention_maps"][i, j]
                    attention_map = img.scale(attention_map, attention_map.min(), attention_map.max(), 0.0, 1.0)
                    attention_map = cv2.resize(attention_map, (256, 256))
                    bounding_box = img.search_bounding_box(attention_map, 0.5)

                    attention_map_image = np.copy(predict_result["images"])
                    attention_map_image += np.pad(np.expand_dims(attention_map, axis=-1), [[0, 0], [0, 0], [0, 2]], "constant")
                    attention_map_images[-1].append(attention_map_image)

                    bounding_box_image = np.copy(predict_result["images"])
                    bounding_box_image = cv2.rectangle(bounding_box_image, bounding_box[0][::-1], bounding_box[1][::-1], (255, 0, 0), 2)
                    bounding_box_images[-1].append(bounding_box_image)

            attention_map_images = np.concatenate([
                np.concatenate(attention_map_images, axis=1)
                for attention_map_images in attention_map_images
            ], axis=0)

            bounding_box_images = np.concatenate([
                np.concatenate(bounding_box_images, axis=1)
                for bounding_box_images in bounding_box_images
            ], axis=0)

            attention_map_images = cv2.cvtColor(attention_map_images, cv2.COLOR_BGR2RGB)
            bounding_box_images = cv2.cvtColor(bounding_box_images, cv2.COLOR_BGR2RGB)

            attention_map_images = img.scale(attention_map_images, 0.0, 1.0, 0.0, 255.0)
            bounding_box_images = img.scale(bounding_box_images, 0.0, 1.0, 0.0, 255.0)

            predictions = "_".join(["".join([class_chars[class_id] for class_id in class_ids]) for class_ids in predict_result["predictions"]])

            cv2.imwrite("outputs/multi_synth/{}_attention_map.jpg".format(predictions), attention_map_images)
            cv2.imwrite("outputs/multi_synth/{}_bounding_box.jpg".format(predictions), bounding_box_images)


if __name__ == "__main__":
    tf.app.run()
