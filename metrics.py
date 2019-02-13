import tensorflow as tf


def dense_to_sparse(tensor, blank):

    indices = tf.where(tf.not_equal(tensor, blank))
    values = tf.gather_nd(tensor, indices)
    shape = tf.shape(tensor, out_type=tf.int64)

    return tf.SparseTensor(indices, values, shape)


def edit_distance(labels, logits, sequence_lengths, normalize, name="edit_distance"):

    predictions = tf.nn.ctc_greedy_decoder(
        inputs=tf.transpose(logits, [1, 0, 2]),
        sequence_length=tf.cast(sequence_lengths, tf.int32),
        merge_repeated=False
    )[0][0]

    labels = dense_to_sparse(labels, logits.shape[-1] - 1)

    return tf.reduce_mean(tf.edit_distance(
        hypothesis=tf.cast(predictions, tf.int32),
        truth=tf.cast(labels, tf.int32),
        normalize=normalize
    ), name=name)


def word_accuracy(labels, predictions, name="word_accuracy"):

    return tf.reduce_mean(tf.cast(tf.reduce_all(tf.equal(
        x=predictions,
        y=labels
    ), axis=1), dtype=tf.float32), name=name)
