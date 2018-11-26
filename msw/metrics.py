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

    return tf.metrics.mean(tf.reduce_all(
        input_tensor=tf.equal(labels, predictions),
        axis=1
    ))


def edit_distance_accuracy(logits, labels, time_major):

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

    labels = dense_to_sparse(
        tensor=labels,
        null=tf.shape(logits)[2] - 1
    )

    return tf.metrics.mean(1.0 - tf.edit_distance(
        hypothesis=tf.cast(predictions, tf.int32),
        truth=tf.cast(labels, tf.int32),
        normalize=False
    ) / tf.cast(tf.shape(logits)[0], tf.float32))
