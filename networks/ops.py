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


def spatial_transformer(inputs, params, out_size, name="spatial_transformer"):
    """Spatial Transformer Layer
    Implements a spatial transformer layer as described in [1]_.
    Based on [2]_ and edited by David Dao for Tensorflow.
    Parameters
    ----------
    inputs : float
        The output of a convolutional net should have the
        shape [num_batch, height, width, num_channels].
    params: float
        The output of the
        localisation network should be [num_batch, 6].
    out_size: tuple of two ints
        The size of the output of the network (height, width)
    References
    ----------
    .. [1]  Spatial Transformer Networks
            Max Jaderberg, Karen Simonyan, Andrew Zisserman, Koray Kavukcuoglu
            Submitted on 5 Jun 2015
    .. [2]  https://github.com/skaae/transformer_network/blob/master/transformerlayer.py
    Notes
    -----
    To initialize the network to the identity transform init
    ``params`` to :
        identity = np.array([[1., 0., 0.],
                             [0., 1., 0.]])
        identity = identity.flatten()
        params = tf.Variable(initial_value=identity)
    """

    def repeat(inputs, num_repeats):
        with tf.variable_scope("repeat"):
            rep = tf.transpose(tf.expand_dims(tf.ones([num_repeats]), 1), [1, 0])
            rep = tf.cast(rep, tf.int32)
            outputs = tf.matmul(tf.reshape(inputs, [-1, 1]), rep)
            outputs = tf.reshape(inputs, [-1])
            return outputs

    def interpolate(inputs, x, y, out_size):
        with tf.variable_scope("interpolate"):
            # constants
            num_batch = tf.shape(inputs)[0]
            height, width, num_channels = inputs.shape[1:]

            x = tf.cast(x, tf.float32)
            y = tf.cast(y, tf.float32)

            # scale indices from [-1, 1] to [0, width/height]
            x = (x + 1.0) * tf.cast(width, tf.float32) / 2.0
            y = (y + 1.0) * tf.cast(height, tf.float32) / 2.0

            # do sampling
            x0 = tf.cast(tf.floor(x), tf.int32)
            x1 = x0 + 1
            y0 = tf.cast(tf.floor(y), tf.int32)
            y1 = y0 + 1

            zero = tf.zeros([], tf.int32)
            max_y = tf.cast(height - 1, tf.int32)
            max_x = tf.cast(width - 1, tf.int32)

            x0 = tf.clip_by_value(x0, zero, max_x)
            x1 = tf.clip_by_value(x1, zero, max_x)
            y0 = tf.clip_by_value(y0, zero, max_y)
            y1 = tf.clip_by_value(y1, zero, max_y)

            dim2 = width
            dim1 = width * height
            base = repeat(tf.range(num_batch) * dim1, out_size[0] * out_size[1])
            base_y0 = base + y0 * dim2
            base_y1 = base + y1 * dim2
            idx_a = base_y0 + x0
            idx_b = base_y1 + x0
            idx_c = base_y0 + x1
            idx_d = base_y1 + x1

            # use indices to lookup pixels in the flat image and restore
            # channels dim
            inputs_flat = tf.reshape(inputs, [-1, num_channels])
            inputs_flat = tf.cast(inputs_flat, tf.float32)
            Ia = tf.gather(inputs_flat, idx_a)
            Ib = tf.gather(inputs_flat, idx_b)
            Ic = tf.gather(inputs_flat, idx_c)
            Id = tf.gather(inputs_flat, idx_d)

            # and finally calculate interpolated values
            x0 = tf.cast(x0, tf.float32)
            x1 = tf.cast(x1, tf.float32)
            y0 = tf.cast(y0, tf.float32)
            y1 = tf.cast(y1, tf.float32)
            wa = tf.expand_dims(((x1 - x) * (y1 - y)), 1)
            wb = tf.expand_dims(((x1 - x) * (y - y0)), 1)
            wc = tf.expand_dims(((x - x0) * (y1 - y)), 1)
            wd = tf.expand_dims(((x - x0) * (y - y0)), 1)

            output = tf.add_n([wa * Ia, wb * Ib, wc * Ic, wd * Id])
            return output

    def meshgrid(height, width):
        with tf.variable_scope("meshgrid"):
            # This should be equivalent to:
            #  x_t, y_t = np.meshgrid(np.linspace(-1, 1, width), np.linspace(-1, 1, height))
            #  ones = np.ones(np.prod(x_t.shape))
            #  grid = np.vstack([x_t.flatten(), y_t.flatten(), ones])
            x_t = tf.matmul(tf.ones([height, 1]), tf.transpose(tf.expand_dims(tf.linspace(-1.0, 1.0, width), 1), [1, 0]))
            y_t = tf.matmul(tf.expand_dims(tf.linspace(-1.0, 1.0, height), 1), tf.ones([1, width]))
            x_t_flat = tf.reshape(x_t, [1, -1])
            y_t_flat = tf.reshape(y_t, [1, -1])

            grid = tf.concat([x_t_flat, y_t_flat, tf.ones_like(x_t_flat)], 0)
            return grid

    def transform(inputs, params, out_size):
        with tf.variable_scope("transform"):
            # constants
            num_batch = tf.shape(inputs)[0]
            height, width, num_channels = inputs.shape[1:]

            params = tf.reshape(params, [-1, 2, 3])
            params = tf.cast(params, tf.float32)

            # grid of (x_t, y_t, 1), eq (1) in ref [1]
            grid = meshgrid(out_size[0], out_size[1])
            grid = tf.expand_dims(grid, 0)
            grid = tf.reshape(grid, [-1])
            grid = tf.tile(grid, [num_batch])
            grid = tf.reshape(grid, [num_batch, 3, -1])

            # Transform A x (x_t, y_t, 1)^T -> (x_s, y_s)
            T_g = tf.matmul(params, grid)
            x_s = tf.slice(T_g, [0, 0, 0], [-1, 1, -1])
            y_s = tf.slice(T_g, [0, 1, 0], [-1, 1, -1])
            x_s_flat = tf.reshape(x_s, [-1])
            y_s_flat = tf.reshape(y_s, [-1])

            outputs = interpolate(inputs, x_s_flat, y_s_flat, out_size)
            outputs = tf.reshape(outputs, [num_batch, out_size[0], out_size[1], num_channels])
            return outputs

    with tf.variable_scope(name):
        outputs = transform(inputs, params, out_size)
        return outputs
