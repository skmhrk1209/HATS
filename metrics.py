import tensorflow as tf


def dense_to_sparse(tensor, null):

    indices = tf.where(tf.not_equal(tensor, null))
    values = tf.gather_nd(tensor, indices)
    shape = tf.shape(tensor, out_type=tf.int64)

    return tf.SparseTensor(indices, values, shape)


def edit_distance_accuracy(logits, labels):

    indices = tf.not_equal(labels, tf.shape(logits)[2] - 1)
    indices = tf.where(tf.reduce_any(indices, axis=1))
    logits = tf.gather_nd(logits, indices)
    labels = tf.gather_nd(labels, indices)

    logits = tf.transpose(logits, [1, 0, 2])

    predictions = tf.nn.ctc_greedy_decoder(
        inputs=logits,
        sequence_length=tf.tile(
            input=[tf.shape(logits)[0]],
            multiples=[tf.shape(logits)[1]]
        ),
        merge_repeated=False
    )[0][0]

    labels = dense_to_sparse(
        tensor=labels,
        null=tf.shape(logits)[2] - 1
    )

    return 1.0 - tf.edit_distance(
        hypothesis=tf.cast(predictions, tf.int32),
        truth=tf.cast(labels, tf.int32),
        normalize=True
    )
