import tensorflow as tf
import numpy as np
import argparse
import itertools
import cv2
from utils.attr_dict import AttrDict
from data.multi_mjsynth import Dataset
from models.acnn import Model
from networks.residual_network import ResidualNetwork
from networks.hierarchical_recurrent_attention_network import HierarchicalRecurrentAttentionNetwork

parser = argparse.ArgumentParser()
parser.add_argument("--model_dir", type=str, default="multi_mjsynth_acnn_model", help="model directory")
parser.add_argument('--filenames', type=str, nargs="+", default=["multi_mjsynth_train.tfrecord"], help="tfrecord filenames")
parser.add_argument("--num_epochs", type=int, default=100, help="number of training epochs")
parser.add_argument("--batch_size", type=int, default=128, help="batch size")
parser.add_argument("--buffer_size", type=int, default=100000, help="buffer size to shuffle dataset")
parser.add_argument("--data_format", type=str, choices=["channels_first", "channels_last"], default="channels_last", help="data_format")
parser.add_argument("--train", action="store_true", help="with training")
parser.add_argument("--eval", action="store_true", help="with evaluation")
parser.add_argument("--predict", action="store_true", help="with prediction")
parser.add_argument("--gpu", type=str, default="0", help="gpu id")
args = parser.parse_args()

tf.logging.set_verbosity(tf.logging.INFO)


def main(unused_argv):

    multi_mjsynth_classifier = tf.estimator.Estimator(
        model_fn=Model(
            convolutional_network=ResidualNetwork(
                conv_param=AttrDict(filters=32, kernel_size=[7, 7], strides=[1, 1]),
                pool_param=None,
                residual_params=[
                    AttrDict(filters=32, strides=[2, 2], blocks=2),
                    AttrDict(filters=64, strides=[2, 2], blocks=2),
                    AttrDict(filters=128, strides=[1, 1], blocks=2),
                    AttrDict(filters=256, strides=[1, 1], blocks=2),
                ],
                num_classes=None,
                data_format=args.data_format
            ),
            attention_network=HierarchicalRecurrentAttentionNetwork(
                conv_params=[
                    AttrDict(filters=4, kernel_size=[9, 9], strides=[2, 2]),
                    AttrDict(filters=4, kernel_size=[9, 9], strides=[2, 2]),
                ],
                deconv_params=[
                    AttrDict(filters=16, kernel_size=[3, 3], strides=[2, 2]),
                    AttrDict(filters=16, kernel_size=[3, 3], strides=[2, 2]),
                ],
                global_bottleneck_units=16,
                local_bottleneck_units=16,
                sequence_length=4,
                string_length=10,
                data_format=args.data_format
            ),
            num_classes=63,
            data_format=args.data_format,
            hyper_params=AttrDict(
                cross_entropy_decay=1e-0,
                attention_map_decay=1e-2,
                total_variation_decay=1e-6
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
                data_format=args.data_format,
                image_size=[128, 128],
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
                data_format=args.data_format,
                image_size=[128, 128],
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
                data_format=args.data_format,
                image_size=[128, 128],
                sequence_length=4,
                string_length=10
            ).get_next()
        )

        for i, predict_result in enumerate(itertools.islice(predict_results, 10)):

            for j in range(4):

                for k in range(10):

                    def scale(input, input_min, input_max, output_min, output_max):
                        return output_min + (input - input_min) / (input_max - input_min) * (output_max - output_min)

                    merged_attention_map = predict_result["merged_attention_maps_{}_{}".format(j, k)]
                    merged_attention_map = scale(merged_attention_map, merged_attention_map.min(), merged_attention_map.max(), 0.0, 1.0)
                    merged_attention_map = cv2.resize(merged_attention_map, (128, 128))

                    image = np.array(predict_result["images"])
                    image[:, :, 0] += merged_attention_map

                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    cv2.imwrite("outputs/image_{}_{}_{}.png".format(i, j, k), image * 255.0)
                    cv2.imwrite("outputs/merged_attention_map_{}_{}_{}.png".format(i, j, k), merged_attention_map * 255.0)


if __name__ == "__main__":
    tf.app.run()
