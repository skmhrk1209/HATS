import tensorflow as tf


def dense_to_sparse(tensor, null):

    indices = tf.where(tf.not_equal(tensor, null))
    values = tf.gather_nd(tensor, indices)
    shape = tf.shape(tensor, out_type=tf.int64)

    return tf.SparseTensor(indices, values, shape)


def edit_distance_accuracy(labels, logits):

    time_step, num_classes = tf.unstack(tf.shape(logits)[1:])
    indices = tf.not_equal(labels, num_classes - 1)
    indices = tf.where(tf.reduce_any(indices, axis=1))
    logits = tf.gather_nd(logits, indices)
    labels = tf.gather_nd(labels, indices)

    batch_size = tf.shape(logits)[0]
    predictions = tf.nn.ctc_greedy_decoder(
        inputs=tf.transpose(logits, [1, 0, 2]),
        sequence_length=tf.tile([time_step], [batch_size]),
        merge_repeated=False
    )[0][0]
    labels = dense_to_sparse(labels, num_classes - 1)

    return 1.0 - tf.edit_distance(
        hypothesis=tf.cast(predictions, tf.int32),
        truth=tf.cast(labels, tf.int32),
        normalize=True
    )
