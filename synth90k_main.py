# =============================================================
# dataset details
# dataset: synth90k
# download: http://www.robots.ox.ac.uk/~vgg/data/text/
# train: 7224612
# val: 802734
# test: 891927
# max num chars: 23
# classes: [0-9A-Z](case-insensitive)
# word accuracy:
# edit distance:
# pretrained model: chars74k classifier
# max steps: 100000 batch size: 128
# =============================================================

import tensorflow as tf
import argparse
from attrdict import AttrDict
from dataset import Dataset
from models.hats import HATS
from networks.attention_network import AttentionNetwork
from networks.pyramid_resnet import PyramidResNet
from algorithms import *

parser = argparse.ArgumentParser()
parser.add_argument("--model_dir", type=str, default="synth90k_hats", help="model directory")
parser.add_argument("--pretrained_model_dir", type=str, default="", help="pretrained model directory")
parser.add_argument('--filenames', type=str, nargs="+", default=["synth90k_train.tfrecord"], help="tfrecord filenames")
parser.add_argument("--num_epochs", type=int, default=None, help="number of training epochs")
parser.add_argument("--batch_size", type=int, default=128, help="batch size")
parser.add_argument("--data_format", type=str, default="channels_first", help="data format")
parser.add_argument("--steps", type=int, default=None, help="number of training epochs")
parser.add_argument("--max_steps", type=int, default=None, help="maximum number of training epochs")
parser.add_argument("--train", action="store_true", help="with training")
parser.add_argument("--eval", action="store_true", help="with evaluation")
parser.add_argument("--predict", action="store_true", help="with prediction")
parser.add_argument("--gpu", type=str, default="0,1,2", help="gpu id")
parser.add_argument("--random_seed", type=int, default=1209, help="random seed")
args = parser.parse_args()

tf.logging.set_verbosity(tf.logging.INFO)


def main(unused_argv):

    classifier = tf.estimator.Estimator(
        model_fn=lambda features, labels, mode: HATS(
            backbone_network=PyramidResNet(
                conv_param=AttrDict(filters=64, kernel_size=[7, 7], strides=[2, 2]),
                pool_param=None,
                residual_params=[
                    AttrDict(filters=64, strides=[2, 2], blocks=2),
                    AttrDict(filters=128, strides=[2, 2], blocks=2),
                    AttrDict(filters=256, strides=[2, 2], blocks=2),
                    AttrDict(filters=512, strides=[2, 2], blocks=2),
                ],
                data_format=args.data_format
            ),
            attention_network=AttentionNetwork(
                conv_params=[
                    AttrDict(filters=8, kernel_size=[3, 3], strides=[2, 2]),
                    AttrDict(filters=8, kernel_size=[3, 3], strides=[2, 2]),
                ],
                rnn_params=[
                    AttrDict(sequence_length=23, units=256),
                ],
                deconv_params=[
                    AttrDict(filters=8, kernel_size=[3, 3], strides=[2, 2]),
                    AttrDict(filters=8, kernel_size=[3, 3], strides=[2, 2]),
                ],
                data_format=args.data_format
            ),
            num_classes=37,
            data_format=args.data_format,
            hyper_params=AttrDict(
                attention_decay_fn=lambda global_step: tf.train.cosine_decay(
                    learning_rate=1e-6,
                    global_step=global_step,
                    decay_steps=args.max_steps
                ),
                learning_rate=0.1,
                decay_steps=args.max_steps
            )
        )(features, labels, mode),
        model_dir=args.model_dir,
        config=tf.estimator.RunConfig(
            tf_random_seed=args.random_seed,
            session_config=tf.ConfigProto(
                gpu_options=tf.GPUOptions(
                    visible_device_list=args.gpu,
                    allow_growth=True
                )
            )
        ),
        warm_start_from=tf.estimator.WarmStartSettings(
            ckpt_to_initialize_from=args.pretrained_model_dir,
            vars_to_warm_start=".*pyramid_resnet.*"
        )
    )

    if args.train:

        classifier.train(
            input_fn=Dataset(
                filenames=args.filenames,
                num_epochs=args.num_epochs,
                batch_size=args.batch_size,
                random_seed=args.random_seed,
                sequence_lengths=[23],
                image_size=[256, 256],
                data_format=args.data_format,
                encoding="jpeg"
            ),
            steps=args.steps,
            max_steps=args.max_steps
        )

    if args.eval:

        eval_results = classifier.evaluate(
            input_fn=Dataset(
                filenames=args.filenames,
                num_epochs=1,
                batch_size=args.batch_size,
                random_seed=args.random_seed,
                sequence_lengths=[23],
                image_size=[256, 256],
                data_format=args.data_format,
                encoding="jpeg"
            )
        )

        print(eval_results)

    if args.predict:

        import cv2
        import itertools

        predict_results = classifier.predict(
            input_fn=Dataset(
                filenames=args.filenames,
                num_epochs=1,
                batch_size=args.batch_size,
                random_seed=args.random_seed,
                sequence_lengths=[23],
                image_size=[256, 256],
                data_format=args.data_format,
                encoding="jpeg"
            )
        )

        for i, predict_result in enumerate(itertools.islice(predict_results, 100)):

            image = predict_result["images"]
            attention_maps = predict_result["attention_maps"]

            if args.data_format == "channels_first":
                image = np.transpose(image, [1, 2, 0])
                attention_maps = np.transpose(attention_maps, [0, 2, 3, 1])

            for attention_maps in attention_maps:

                attention_map = (attention_map - attention_map.min()) / (attention_map.max() - attention_map.min())
                attention_map[attention_map < 0.5] = 0.0
                attention_map = cv2.resize(attention_map, image.shape[:-1])
                image[:, :, -1] += attention_map

            cv2.imwrite("outputs/synth90k/attention_map_{}.jpg".format(i), image * 255.)


if __name__ == "__main__":
    tf.app.run()
