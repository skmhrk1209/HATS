import tensorflow as tf
import numpy as np
import cv2
import argparse
from attrdict import AttrDict
from icdar2015_word_recognition.dataset import Dataset
from model import Model
from networks.residual_network import ResidualNetwork
from networks.attention_network import AttentionNetwork

parser = argparse.ArgumentParser()
parser.add_argument("--model_dir", type=str, default="icdar2015_word_recognition_acnn_model", help="model directory")
parser.add_argument('--filenames', type=str, nargs="+", default=["icdar2015_word_recognition_train.tfrecord"], help="tfrecord filenames")
parser.add_argument("--num_epochs", type=int, default=1000, help="number of training epochs")
parser.add_argument("--batch_size", type=int, default=128, help="batch size")
parser.add_argument("--buffer_size", type=int, default=4468, help="buffer size to shuffle dataset")
parser.add_argument("--data_format", type=str, default="channels_first", help="data format")
parser.add_argument("--train", action="store_true", help="with training")
parser.add_argument("--eval", action="store_true", help="with evaluation")
parser.add_argument("--predict", action="store_true", help="with prediction")
parser.add_argument("--gpu", type=str, default="0,1,2", help="gpu id")
args = parser.parse_args()

tf.logging.set_verbosity(tf.logging.INFO)


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
                rnn_params=[
                    AttrDict(sequence_length=21, num_units=256),
                ],
                data_format=args.data_format
            ),
            num_classes=96,
            data_format=args.data_format,
            accuracy_type=Model.AccuracyType.EDIT_DISTANCE,
            hyper_params=AttrDict(
                learning_rate=0.001,
                beta1=0.9,
                beta2=0.999,
                attention_map_decay=0.0001
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
                sequence_lengths=[21],
                image_size=[256, 256],
                data_format=args.data_format
            ).get_next()
        )

    if args.predict:

        filenames = []

        with tf.Session() as session:

            next_filename = tf.data.TFRecordDataset(args.filenames).map(lambda example: tf.parse_single_example(
                serialized=example,
                features=dict(path=tf.FixedLenFeature(shape=[], dtype=tf.string))
            )["path"]).make_one_shot_iterator().get_next()

            while True:
                try:
                    filenames.append(session.run(next_filename))
                    print(filenames[-1])
                except:
                    break

        images = list(map(lambda filename: np.transpose(
            cv2.resize(cv2.imread(str(filename)), (256, 256)),
            [2, 0, 1] if args.data_format == "channels_first" else [0, 1, 2]
        ), filenames))

        predict_results = classifier.predict(
            input_fn=tf.estimator.inputs.numpy_input_fn(
                x={"image": images},
                batch_size=args.batch_size,
                num_epochs=1,
                shuffle=False
            )
        )

        with open("result.txt", "w") as f:

            for filename, predict_result in zip(filenames, predict_results):
                f.write('{}, "{}"'.format(filename, "".join(map(lambda x: chr(x + 32), filter(lambda x: x < 95, predict_result["predictions"])))))


if __name__ == "__main__":
    tf.app.run()
