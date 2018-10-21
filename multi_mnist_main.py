import tensorflow as tf
import numpy as np
import argparse
import itertools
import seaborn
import matplotlib.pyplot as plt
from utils.attr_dict import AttrDict
from models.acnn import Model
from networks.residual_network import ResidualNetwork
from networks.recurrent_attention_network import RecurrentAttentionNetwork

parser = argparse.ArgumentParser()
parser.add_argument("--model_dir", type=str, default="multi_mnist_acnn_model", help="model directory")
parser.add_argument('--filenames', type=str, nargs="+", default=["multi_mnist_train.tfrecord"], help="tfrecord filenames")
parser.add_argument("--num_epochs", type=int, default=100, help="number of training epochs")
parser.add_argument("--batch_size", type=int, default=128, help="batch size")
parser.add_argument("--buffer_size", type=int, default=50000, help="buffer size to shuffle dataset")
parser.add_argument("--data_format", type=str, choices=["channels_first", "channels_last"], default="channels_last", help="data_format")
parser.add_argument("--train", action="store_true", help="with training")
parser.add_argument("--eval", action="store_true", help="with evaluation")
parser.add_argument("--predict", action="store_true", help="with prediction")
parser.add_argument("--gpu", type=str, default="0", help="gpu id")
args = parser.parse_args()

tf.logging.set_verbosity(tf.logging.INFO)


def main(unused_argv):

    def make_multi_mnist(images, labels, image_size, min_occurrences, max_occurrences):

        multi_images = []
        multi_labels = []

        for _ in range(images.shape[0]):

            num_occurrences = np.random.random_integers(min_occurrences, max_occurrences)
            indices = np.random.randint(images.shape[0], size=num_occurrences)
            translations = []

            while len(translations) < len(indices):

                yy = np.random.randint(image_size[0] - 28)
                xx = np.random.randint(image_size[1] - 28)

                for y, x in translations:
                    if y - 28 < yy < y + 28 and x - 28 < xx < x + 28:
                        break
                else:
                    translations.append([yy, xx])

            multi_image = np.zeros(image_size + [images.shape[-1]], np.float32)
            multi_label = np.pad(labels[indices], [0, max_occurrences - len(indices)], "constant", constant_values=10)
            multi_label = np.expand_dims(multi_label, axis=-1)

            for image, (y, x) in zip(images[indices], sorted(translations)):

                multi_image[y: y + 28, x: x + 28, :] += image

            multi_images.append(multi_image)
            multi_labels.append(multi_label)

        return np.array(multi_images), np.array(multi_labels)

    mnist = tf.contrib.learn.datasets.load_dataset("mnist")
    train_images = mnist.train.images.astype(np.float32).reshape([-1, 28, 28, 1])
    train_labels = mnist.train.labels.astype(np.int32)
    test_images = mnist.test.images.astype(np.float32).reshape([-1, 28, 28, 1])
    test_labels = mnist.test.labels.astype(np.int32)

    train_images, train_labels = make_multi_mnist(train_images, train_labels, [128, 128], 1, 4)
    test_images, test_labels = make_multi_mnist(test_images, test_labels, [128, 128], 1, 4)

    imagenet_classifier = tf.estimator.Estimator(
        model_fn=Model(
            convolutional_network=ResidualNetwork(
                conv_param=AttrDict(filters=32, kernel_size=[3, 3], strides=[1, 1]),
                pool_param=None,
                residual_params=[
                    AttrDict(filters=32, strides=[2, 2], blocks=1),
                    AttrDict(filters=64, strides=[2, 2], blocks=1)
                ],
                num_classes=None,
                data_format=args.data_format
            ),
            recurrent_attention_network=RecurrentAttentionNetwork(
                conv_params=[
                    AttrDict(filters=4, kernel_size=[9, 9], strides=[2, 2]),
                    AttrDict(filters=4, kernel_size=[9, 9], strides=[2, 2]),
                ],
                deconv_params=[
                    AttrDict(filters=16, kernel_size=[3, 3], strides=[2, 2]),
                    AttrDict(filters=16, kernel_size=[3, 3], strides=[2, 2]),
                ],
                bottleneck_units=16,
                sequence_length=4,
                data_format=args.data_format
            ),
            num_classes=11,
            num_digits=1,
            data_format=args.data_format,
            hyper_params=AttrDict(
                cross_entropy_decay=1e-0,
                attention_map_decay=1e-2,
                total_variation_decay=1e-5 #1e-6
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

        imagenet_classifier.train(
            input_fn=tf.estimator.inputs.numpy_input_fn(
                x={"images": train_images},
                y=train_labels,
                batch_size=args.batch_size,
                num_epochs=args.num_epochs,
                shuffle=True
            )
        )

    if args.eval:

        eval_results = imagenet_classifier.evaluate(
            input_fn=tf.estimator.inputs.numpy_input_fn(
                x={"images": test_images},
                y=test_labels
            )
        )

        print(eval_results)

    if args.predict:

        predict_results = imagenet_classifier.predict(
            input_fn=tf.estimator.inputs.numpy_input_fn(
                x={"images": test_images},
                y=test_labels
            )
        )

        for i, predict_result in enumerate(itertools.islice(predict_results, 10)):

            reduced_attention_map = predict_result["reduced_attention_maps"]
            seaborn.heatmap(reduced_attention_map.reshape([32, 32]))

            plt.savefig("output/reduced_attention_map_{}.png".format(i))


if __name__ == "__main__":
    tf.app.run()
