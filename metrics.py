import tensorflow as tf


def dense_to_sparse(tensor, blank):

    indices = tf.where(tf.not_equal(tensor, blank))
    values = tf.gather_nd(tensor, indices)
    shape = tf.shape(tensor, out_type=tf.int64)

    return tf.SparseTensor(indices, values, shape)


def edit_distance(labels, logits, sequence_lengths, normalize):

    predictions = tf.nn.ctc_greedy_decoder(
        inputs=tf.transpose(logits, [1, 0, 2]),
        sequence_length=sequence_lengths,
        merge_repeated=False
    )[0][0]

    labels = dense_to_sparse(labels, logits.shape[-1] - 1)

    return tf.metrics.mean(tf.edit_distance(
        hypothesis=tf.cast(predictions, tf.int32),
        truth=tf.cast(labels, tf.int32),
        normalize=normalize
    ))


def word_accuracy(labels, predictions):

    return tf.metrics.mean(tf.reduce_all(tf.equal(predictions, labels), axis=1))
