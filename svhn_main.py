import tensorflow as tf
import numpy as np
import argparse
import itertools
import seaborn
import matplotlib.pyplot as plt
from matplotlib import colors
from utils.attr_dict import AttrDict
from data.svhn import Dataset
from models.svhn_acnn import Model
from networks.residual_network import ResidualNetwork
from networks.attention_network import AttentionNetwork

parser = argparse.ArgumentParser()
parser.add_argument("--model_dir", type=str, default="svhn_acnn_model", help="model directory")
parser.add_argument('--filenames', type=str, nargs="+", default=["train.tfrecord"], help="tfrecord filenames")
parser.add_argument("--num_epochs", type=int, default=100, help="number of training epochs")
parser.add_argument("--batch_size", type=int, default=128, help="batch size")
parser.add_argument("--buffer_size", type=int, default=30000, help="buffer size to shuffle dataset")
parser.add_argument("--data_format", type=str, choices=["channels_first", "channels_last"], default="channels_last", help="data_format")
parser.add_argument("--train", action="store_true", help="with training")
parser.add_argument("--eval", action="store_true", help="with evaluation")
parser.add_argument("--predict", action="store_true", help="with prediction")
parser.add_argument("--gpu", type=str, default="0", help="gpu id")
args = parser.parse_args()

tf.logging.set_verbosity(tf.logging.INFO)


def main(unused_argv):

    imagenet_classifier = tf.estimator.Estimator(
        model_fn=Model(
            convolutional_network=ResidualNetwork(
                conv_param=AttrDict(filters=32, kernel_size=[3, 3], strides=[2, 2]),
                pool_param=None,
                residual_params=[
                    AttrDict(filters=32, strides=[1, 1], blocks=2),
                    AttrDict(filters=64, strides=[1, 1], blocks=2),
                    AttrDict(filters=128, strides=[1, 1], blocks=2),
                    AttrDict(filters=256, strides=[1, 1], blocks=2),
                ],
                num_classes=None,
                data_format=args.data_format
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
                bottleneck_units=16,
                data_format=args.data_format
            ),
            num_classes=11,
            num_digits=4,
            data_format=args.data_format,
            hyper_params=AttrDict(
                attention_map_decay=1e-2,
                total_variation_decay=1e-4
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
            input_fn=lambda: Dataset(
                filenames=args.filenames,
                num_epochs=args.num_epochs,
                batch_size=args.batch_size,
                buffer_size=args.buffer_size,
                data_format=args.data_format,
                image_size=[64, 64]
            ).get_next(),
            hooks=[
                tf.train.LoggingTensorHook(
                    tensors={"softmax": "softmax"},
                    every_n_iter=100
                )
            ]
        )

    if args.eval:

        eval_results = imagenet_classifier.evaluate(
            input_fn=lambda: Dataset(
                filenames=args.filenames,
                num_epochs=args.num_epochs,
                batch_size=args.batch_size,
                buffer_size=args.buffer_size,
                data_format=args.data_format,
                image_size=[64, 64]
            ).get_next()
        )

        print(eval_results)

    if args.predict:

        predict_results = imagenet_classifier.predict(
            input_fn=lambda: Dataset(
                filenames=args.filenames,
                num_epochs=args.num_epochs,
                batch_size=args.batch_size,
                buffer_size=args.buffer_size,
                data_format=args.data_format,
                image_size=[64, 64]
            ).get_next()
        )

        fig, axs = plt.subplots(2, 4)
        fig.suptitle('Multiple images')

        images = []
        for i, predict_result in enumerate(itertools.islice(predict_results, 10)):
            images.append(axs[i, j].imshow(predict_result["features"], cmap="cool"))
            images.append(axs[i, j].imshow(predict_result["reduced_attention_maps"], cmap="cool"))

        # Find the min and max of all colors for use in setting the color scale.
        vmin = min(image.get_array().min() for image in images)
        vmax = max(image.get_array().max() for image in images)
        norm = colors.Normalize(vmin=vmin, vmax=vmax)
        for im in images:
            im.set_norm(norm)

        fig.colorbar(images[0], ax=axs, orientation='horizontal', fraction=.1)


        # Make images respond to changes in the norm of other images (e.g. via the
        # "edit axis, curves and images parameters" GUI on Qt), but be careful not to
        # recurse infinitely!
        def update(changed_image):
            for im in images:
                if (changed_image.get_cmap() != im.get_cmap()
                        or changed_image.get_clim() != im.get_clim()):
                    im.set_cmap(changed_image.get_cmap())
                    im.set_clim(changed_image.get_clim())


        for im in images:
            im.callbacksSM.connect('changed', update)

        plt.show()

if __name__ == "__main__":
    tf.app.run()
