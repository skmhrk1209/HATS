import tensorflow as tf

with tf.Session() as sess:

    label = [1, 2, 0, 3, 4, 0, 5, 6, 133, 133]
    label = tf.cast(label, tf.int32)
    label = tf.gather(label, tf.where(tf.not_equal(label, 133)))
    label = tf.reshape(label, [-1])
    last_indices = tf.where(tf.equal(label, 0))
    last_indices = tf.reshape(last_indices, [-1])
    first_indices = tf.concat([[0], last_indices[:-1] + 1], axis=0)
    indices = tf.stack([first_indices, last_indices], axis=-1)
    label = tf.map_fn(lambda indices: label[indices[0], indices[1]], indices)
    print(sess.run(label))