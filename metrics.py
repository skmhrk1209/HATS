import tensorflow as tf


def dense_to_sparse(tensor, null):

    indices = tf.where(tf.not_equal(tensor, null))
    values = tf.gather_nd(tensor, indices)
    shape = tf.shape(tensor, out_type=tf.int64)

    return tf.SparseTensor(indices, values, shape)


def edit_distance(labels, logits, normalize):

    num_classes = tf.shape(logits)[2]
    indices = tf.not_equal(labels, num_classes - 1)
    indices = tf.where(tf.reduce_any(indices, axis=1))
    labels = tf.gather_nd(labels, indices)
    logits = tf.gather_nd(logits, indices)

    labels = dense_to_sparse(labels, num_classes - 1)

    batch_size, time_step = tf.unstack(tf.shape(logits)[:2])
    predictions = tf.nn.ctc_greedy_decoder(
        inputs=tf.transpose(logits, [1, 0, 2]),
        sequence_length=tf.tile([time_step], [batch_size]),
        merge_repeated=False
    )[0][0]

    return tf.metrics.mean(tf.edit_distance(
        hypothesis=tf.cast(predictions, tf.int32),
        truth=tf.cast(labels, tf.int32),
        normalize=normalize
    ))


def sequence_accuracy(labels, logits):

    num_classes = tf.shape(logits)[2]
    indices = tf.not_equal(labels, num_classes - 1)
    indices = tf.where(tf.reduce_any(indices, axis=1))
    labels = tf.gather_nd(labels, indices)
    logits = tf.gather_nd(logits, indices)

    predictions = tf.argmax(logits, axis=2, output_type=tf.int32)

    return tf.metrics.mean(tf.reduce_all(tf.equal(labels, predictions), axis=1))
