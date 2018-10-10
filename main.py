import tensorflow as tf
import tensorflow_hub as hub
import argparse
import functools
from models import acnn
from data import imagenet
from networks.attention_network import AttentionNetwork
from utils.attr_dict import AttrDict

parser = argparse.ArgumentParser()
parser.add_argument("--model_dir", type=str, default="imagenet_acnn_model", help="model directory")
parser.add_argument("--filenames", type=str, nargs="+", default=["train.tfrecord"], help="tfrecord filenames")
parser.add_argument("--num_epochs", type=int, default=1, help="number of training epochs")
parser.add_argument("--batch_size", type=int, default=100, help="batch size")
parser.add_argument("--buffer_size", type=int, default=1000000, help="buffer size to shuffle dataset")
parser.add_argument('--data_format', type=str, choices=["channels_first", "channels_last"], default="channels_last", help="data_format")
parser.add_argument('--train', action="store_true", help="training mode")
parser.add_argument('--eval', action="store_true", help="evaluation mode")
parser.add_argument('--predict', action="store_true", help="prediction mode")
parser.add_argument('--gpu', type=str, default="0", help="gpu id")
args = parser.parse_args()

tf.logging.set_verbosity(tf.logging.INFO)

resnet_v2_50 = "https://tfhub.dev/google/imagenet/resnet_v2_50/feature_vector/1"

with tf.device('/device:GPU:0'):


    def main(unused_argv):

        imagenet_classifier = tf.estimator.Estimator(
            model_fn=acnn.Model(
                convolutional_network=lambda inputs: hub.Module(resnet_v2_50)(
                    dict(images=inputs),
                    signature="image_feature_vector",
                    as_dict=True
                )["resnet_v2_50/block4"],
                attention_network=AttentionNetwork(
                    conv_params=[AttrDict(
                        filters=4,
                        kernel_size=9,
                        strides=1
                    )] * 2,
                    deconv_params=[AttrDict(
                        filters=16,
                        kernel_size=3,
                        strides=1
                    )] * 2,
                    bottleneck_units=128,
                    data_format="channels_last"
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
            ),
            params=dict(
                attention_decay=1e-6
            )
        )

        if args.train:

            logging_hook = tf.train.LoggingTensorHook(
                tensors={
                    "softmax": "softmax"
                },
                every_n_iter=100
            )

            imagenet_classifier.train(
                input_fn=lambda: imagenet.Dataset(
                    image_size=[224, 224],
                    data_format="channels_last",
                    filenames=args.filenames,
                    num_epochs=args.num_epochs,
                    batch_size=args.batch_size,
                    buffer_size=args.buffer_size
                ).get_next(),
                hooks=[logging_hook]
            )

        if args.eval:

            eval_results = imagenet_classifier.evaluate(
                input_fn=lambda: imagenet.Dataset(
                    image_size=[224, 224],
                    data_format="channels_last",
                    filenames=args.filenames,
                    num_epochs=args.num_epochs,
                    batch_size=args.batch_size,
                    buffer_size=args.buffer_size
                ).get_next()
            )

            print(eval_results)


    if __name__ == "__main__":
        tf.app.run()
