import tensorflow as tf


def dense_to_sparse(tensor, null):

    indices = tf.where(tf.not_equal(tensor, null))
    values = tf.gather_nd(tensor, indices)
    shape = tf.shape(tensor, out_type=tf.int64)

    return tf.SparseTensor(indices, values, shape)


def full_sequence_accuracy(logits, labels, time_major):

    if time_major:
        logits = tf.transpose(labels, [1, 0, 2])
        labels = tf.transpose(labels, [1, 0])

    predictions = tf.argmax(
        input=logits,
        axis=2,
        output_type=tf.int32
    )

    return tf.reduce_all(tf.equal(labels, predictions), axis=1)


def edit_distance_accuracy(logits, labels, time_major):

    if time_major:
        logits = tf.transpose(logits, [1, 0, 2])
        labels = tf.transpose(labels, [1, 0])

    print(logits.shape)

    indices = tf.not_equal(labels, tf.shape(logits)[2] - 1)
    indices = tf.where(tf.reduce_any(indices, axis=1))
    logits = tf.gather(logits, indices)
    labels = tf.gather(labels, indices)

    print(indices.shape)

    print(logits.shape)

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
        normalize=False
    ) / tf.cast(tf.shape(logits)[0], tf.float32)
