import tensorflow as tf

x1 = tf.zeros([10, 10])
x2 = tf.zeros([10, 10])
c = tf.nn.rnn_cell.LSTMCell(10)

y1 = tf.nn.static_rnn(c, [x1], dtype=tf.float32)
y2 = tf.nn.static_rnn(c, [x2], dtype=tf.float32)

with tf.Session() as sess:

    sess.run(tf.global_variables_initializer())

    print(tf.trainable_variables())

    sess.run([y1, y2])