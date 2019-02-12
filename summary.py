import tensorflow as tf
import re


def any(tensor, data_format=None, **kwargs):

    name = re.sub(":.*", "", tensor.name)

    if len(tensor.shape) == 1:

        tf.summary.scalar(name, tensor, **kwargs)

    if len(tensor.shape) == 4:

        if data_format and data_format == "channels_first":
            tensor = tf.transpose(tensor, [0, 2, 3, 1])

        tf.summary.image(name, tensor, **kwargs)
