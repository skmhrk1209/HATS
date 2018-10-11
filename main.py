import tensorflow as tf
import argparse
from models import acnn
from networks.attention_network import AttentionNetwork
from networks.classification_network import ClassificationNetwork
from data import imagenet
from utils.attr_dict import AttrDict

parser = argparse.ArgumentParser()
parser.add_argument("--model_dir", type=str, default="imagenet_acnn_model", help="model directory")
parser.add_argument('--filenames', type=str, nargs="+", default=["train.tfrecord"], help="tfrecord filenames")
parser.add_argument("--num_epochs", type=int, default=1, help="number of training epochs")
parser.add_argument("--batch_size", type=int, default=128, help="batch size")
parser.add_argument("--buffer_size", type=int, default=1000000, help="buffer size to shuffle dataset")
parser.add_argument('--data_format', type=str, choices=["channels_first", "channels_last"], default="channels_last", help="data_format")
parser.add_argument('--train', action="store_true", help="with training")
parser.add_argument('--gpu', type=str, default="0", help="gpu id")
args = parser.parse_args()

tf.logging.set_verbosity(tf.logging.INFO)
tf.keras.backend.set_learning_phase(0)

config = tf.ConfigProto(
    gpu_options=tf.GPUOptions(
        visible_device_list=args.gpu,
        allow_growth=True
    ),
    log_device_placement=False,
    allow_soft_placement=True
)

with tf.Session(config=config) as session:

    acnn_model = acnn.Model(
        dataset=imagenet.Dataset(),
        convolutional_network=tf.keras.applications.resnet50.ResNet50(
            include_top=False,
            weights="imagenet",
            pooling=None
        ),
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
        ),
        classification_network=ClassificationNetwork(
            dense_params=[AttrDict(
                units=1024
            )],
            num_classes=1000,
            data_format="channels_last"
        ),
        hyper_parameters=AttrDict(
            attention_decay=1e-6,
            learning_rate=0.001,
            beta1=0.9,
            beta2=0.999
        ),
        name=args.model_dir
    )

    acnn_model.initialize()

    if args.train:

        acnn_model.train(
            filenames=args.filenames,
            num_epochs=args.num_epochs,
            batch_size=args.batch_size,
            buffer_size=args.buffer_size
        )
