import tensorflow as tf

with tf.Session() as sess:

    label = [1, 2, 0, 3, 4, 0, 5, 6, 133, 133]
    indices = tf.cast(tf.squeeze(tf.where(tf.not_equal(label, 133))), tf.int32)
    label = tf.gather(label, indices)
    label = tf.concat([[0], label, [0]], axis=0)
    indices = tf.cast(tf.squeeze(tf.where(tf.equal(label, 0))), tf.int32)
    indices = tf.stack([indices[:-1], indices[1:]], axis=-1)
    label = tf.map_fn(
        fn=lambda indices: tf.pad(
            tensor=label[indices[0] + 1: indices[1]],
            paddings=[0, 4 - (indices[1] - (indices[0] + 1))],
            mode="constant",
            constant_values=133
        ), 
        elems=indices
    )
    print(sess.run(label))
