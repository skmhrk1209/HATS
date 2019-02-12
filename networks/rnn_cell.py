import tensorflow as tf


class IRNNCell(object):

    def __init__(self, input_units, hidden_units, output_units,
                 kernel_initializer, bias_initializer,
                 name="irnn_cell", reuse=None):

        with tf.variable_scope(name=name, reuse=reuse):

            self.input_kernel = tf.get_variable(
                name="input_kernel",
                shape=[input_units, hidden_units],
                initializer=kernel_initializer,
                trainable=True
            )
            self.input_bias = tf.get_variable(
                name="input_bias",
                shape=[hidden_units],
                initializer=bias_initializer,
                trainable=True
            )
            self.hidden_kernel = tf.get_variable(
                name="hidden_kernel",
                shape=[hidden_units, hidden_units],
                initializer=tf.initializers.identity(),
                trainable=True
            )
            self.hidden_bias = tf.get_variable(
                name="hidden_bias",
                shape=[hidden_units],
                initializer=tf.initializers.zeros(),
                trainable=True
            )
            self.output_kernel = tf.get_variable(
                name="output_kernel",
                shape=[hidden_units, output_units],
                initializer=kernel_initializer,
                trainable=True
            )
            self.output_bias = tf.get_variable(
                name="output_bias",
                shape=[output_units],
                initializer=bias_initializer,
                trainable=True
            )

    def __call__(self, inputs, hiddens):

        hiddens = tf.matmul(hiddens, self.hidden_kernel)
        hiddens += self.hidden_bias
        hiddens += tf.matmul(inputs, self.input_kernel)
        hiddens += self.input_bias
        hiddens = tf.nn.relu(hiddens)

        outputs = tf.matmul(hiddens, self.output_kernel)
        outputs += self.output_bias
        outputs = tf.nn.relu(outputs)

        return outputs, hiddens
