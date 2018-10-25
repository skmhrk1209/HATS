import tensorflow as tf

x1 = tf.zeros([10, 10])
c1 = tf.nn.rnn_cell.LSTMCell(10)

y1 = tf.nn.static_rnn(c1, [x1], dtype=tf.float32, scope="a")
y2 = tf.nn.static_rnn(c1, [x1], dtype=tf.float32, scope="a")

with tf.Session() as sess:

    sess.run(tf.global_variables_initializer())

    print(tf.trainable_variables())

    sess.run([y1, y2])
