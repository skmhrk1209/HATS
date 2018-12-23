import tensorflow as tf


def batch_normalization(inputs, data_format, training, name=None, reuse=None):

    return tf.layers.batch_normalization(
        inputs=inputs,
        axis=1 if data_format == "channels_first" else 3,
        training=training,
        name=name,
        reuse=reuse
    )


def global_average_pooling2d(inputs, data_format):

    return tf.reduce_mean(
        input_tensor=inputs,
        axis=[2, 3] if data_format == "channels_first" else [1, 2]
    )
