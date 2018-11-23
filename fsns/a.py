import tensorflow as tf

with tf.Session() as sess:

    string_length = 4
    sequence_length = 4

    label = [1, 2, 0, 3, 4, 0, 5, 6, 133, 133]
    indices = tf.cast(tf.squeeze(tf.where(tf.not_equal(label, 133))), tf.int32)
    label = tf.gather(label, indices)
    label = tf.concat([[0], label, [0]], axis=0)
    indices = tf.cast(tf.squeeze(tf.where(tf.equal(label, 0))), tf.int32)
    indices = tf.stack([indices[:-1] + 1, indices[1:]], axis=-1)

    label = tf.map_fn(
        fn=lambda range: (lambda label: tf.pad(
            tensor=label,
            paddings=[[0, string_length - tf.shape(label)[0]]],
            mode="constant",
            constant_values=133
        ))(label[range[0]: range[1]]),
        elems=indices
    )

    label = (lambda label: tf.pad(
        tensor=label,
        paddings=[[0, sequence_length - tf.shape(label)[0]], [0, 0]],
        mode="constant",
        constant_values=133
    ))(label)
    print(sess.run(label))
