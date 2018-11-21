import tensorflow as tf


def dense_to_sparse(tensor, blank):

    indices = tf.where(tf.not_equal(tensor, blank))
    values = tf.gather_nd(tensor, indices)
    shape = tf.shape(tensor, out_type=tf.int64)

    return tf.SparseTensor(indices, values, shape)


def accuracy(logits, labels, time_major=True):

    if time_major:
        labels = tf.transpose(labels, [1, 0])
    else:
        logits = tf.transpose(logits, [1, 0, 2])

    predictions = tf.nn.ctc_greedy_decoder(
        inputs=logits,
        sequence_length=tf.tile(
            input=[tf.shape(logits)[0]],
            multiples=[tf.shape(logits)[1]]
        ),
        merge_repeated=False
    )[0][0]

    return tf.metrics.mean(tf.reduce_mean(1.0 - tf.edit_distance(
        hypothesis=tf.cast(predictions, tf.int32),
        truth=dense_to_sparse(labels, tf.shape(logits)[2] - 1),
        normalize=False
    ) / tf.cast(tf.shape(labels)[1], tf.float32)))
