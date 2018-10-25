import tensorflow as tf

x = tf.zeros([10, 10])
c = tf.nn.rnn_cell.LSTMCell(10)

y1 = tf.nn.static_rnn(c, [x], dtype=tf.float32)
y2 = tf.nn.static_rnn(c, [x], dtype=tf.float32)

with tf.Session() as sess:

    sess.run(tf.global_variables_initializer())

    sess.run([y1, y2])