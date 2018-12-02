import tensorflow as tf
import numpy as np


def static_shape(inputs):

    return inputs.get_shape().as_list()


def channels_first(data_format):

    return data_format == "channels_first"


def spatial_flatten(inputs, data_format):

    inputs_shape = static_shape(inputs)
    assert(len(inputs_shape) == 4)

    outputs_shape = ([-1, inputs_shape[1], np.prod(inputs_shape[2:])] if channels_first(data_format) else
                     [-1, np.prod(inputs_shape[1:-1]), inputs_shape[-1]])

    return tf.reshape(inputs, outputs_shape)


def spatial_unflatten(inputs, spatial_shape, data_format):

    inputs_shape = static_shape(inputs)
    assert(len(inputs_shape) == 3)

    outputs_shape = ([-1, inputs_shape[1], spatial_shape[0], spatial_shape[1]] if channels_first(data_format) else
                     [-1, spatial_shape[0], spatial_shape[1], inputs_shape[-1]])

    return tf.reshape(inputs, outputs_shape)


def spatial_softmax(inputs, data_format):

    inputs_shape = static_shape(inputs)

    return spatial_unflatten(
        inputs=tf.nn.softmax(
            logits=spatial_flatten(inputs, data_format),
            dim=2 if channels_first(data_format) else 1
        ),
        spatial_shape=inputs_shape[2:] if channels_first(data_format) else inputs_shape[1:-1],
        data_format=data_format
    )
