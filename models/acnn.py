import tensorflow as tf
import numpy as np
import itertools
import time
import os


class Model(object):

    def __init__(self, dataset, convolutional_network, attention_network, classification_network,
                 hyper_parameters, name="acnn", reuse=None):

        with tf.variable_scope(name, reuse=reuse):

            self.name = name
            self.dataset = dataset
            self.convolutional_network = convolutional_network
            self.attention_network = attention_network
            self.classification_network = classification_network
            self.hyper_parameters = hyper_parameters

            self.images, self.labels = self.dataset.get_next()
            self.training = tf.placeholder(dtype=tf.bool, shape=[])

            self.feature_maps = self.convolutional_network(
                inputs=self.images
            )

            self.attention_maps = self.attention_network(
                inputs=self.feature_maps,
                training=self.training
            )

            shape = self.feature_maps.shape.as_list()
            feature_maps = tf.reshape(
                tensor=self.feature_maps,
                shape=[-1, np.prod(shape[1:3]), shape[3]]
            )

            shape = self.attention_maps.shape.as_list()
            attention_maps = tf.reshape(
                tensor=self.attention_maps,
                shape=[-1, np.prod(shape[1:3]), shape[3]]
            )

            feature_vectors = tf.matmul(
                a=feature_maps,
                b=attention_maps,
                transpose_a=True,
                transpose_b=False
            )

            feature_vectors = tf.layers.flatten(feature_vectors)

            self.logits = self.classification_network(
                inputs=feature_vectors,
                training=self.training
            )

            self.optimizer = tf.train.AdamOptimizer(
                learning_rate=self.hyper_parameters.learning_rate,
                beta1=self.hyper_parameters.beta1,
                beta2=self.hyper_parameters.beta2
            )

            self.loss = tf.losses.sparse_softmax_cross_entropy(
                labels=self.labels,
                logits=self.logits
            )

            self.loss += tf.reduce_mean(tf.reduce_sum(
                input_tensor=tf.abs(self.attention_maps),
                axis=[1, 2, 3]
            )) * self.hyper_parameters.attention_decay

            self.global_step = tf.get_variable(
                name="global_step",
                shape=[],
                dtype=tf.int32,
                initializer=tf.zeros_initializer(),
                trainable=False
            )

            self.trainable_variables = [
                variable for variable in tf.trainable_variables(scope=self.name)
                if not hasattr(variable, "_keras_initialized") or not variable._keras_initialized
            ]

            with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):

                self.train_op = self.optimizer.minimize(
                    loss=self.loss,
                    global_step=self.global_step,
                    var_list=self.trainable_variables
                )

            self.saver = tf.train.Saver()

            attention_maps = tf.reduce_sum(self.attention_maps, axis=3, keepdims=True)
            self.summary = tf.summary.merge([
                tf.summary.image("images", self.images, max_outputs=10),
                tf.summary.image("attention_maps", attention_maps, max_outputs=10),
                tf.summary.scalar("loss", self.loss)
            ])

    def initialize(self):

        session = tf.get_default_session()
        checkpoint = tf.train.latest_checkpoint(self.name)

        if checkpoint:
            self.saver.restore(session, checkpoint)
            print(checkpoint, "loaded")

        else:
            self.global_variables = [
                variable for variable in tf.global_variables(scope=self.name)
                if not hasattr(variable, "_keras_initialized") or not variable._keras_initialized
            ]
            session.run(tf.variables_initializer(self.global_variables))
            print("global variables in {} initialized".format(self.name))

    def train(self, filenames, num_epochs, batch_size, buffer_size):

        session = tf.get_default_session()
        writer = tf.summary.FileWriter(self.name, session.graph)

        print("training started")

        start = time.time()

        self.dataset.initialize(
            filenames=filenames,
            num_epochs=num_epochs,
            batch_size=batch_size,
            buffer_size=buffer_size
        )

        feed_dict = {self.training: True}

        for i in itertools.count():

            try:
                _, global_step = session.run(
                    [self.train_op, self.global_step],
                    feed_dict=feed_dict
                )

            except tf.errors.OutOfRangeError:
                print("training ended")
                break

            if global_step % 100 == 0:

                loss = session.run(self.loss, feed_dict=feed_dict)
                print("global_step: {}, loss: {:.2f}".format(global_step, loss))

                summary = session.run(self.summary, feed_dict=feed_dict)
                writer.add_summary(summary, global_step=global_step)

                if global_step % 10000 == 0:

                    checkpoint = self.saver.save(
                        sess=session,
                        save_path=os.path.join(self.name, "model.ckpt"),
                        global_step=global_step
                    )

                    stop = time.time()
                    print("{} saved ({:.2f} sec)".format(checkpoint, stop - start))
                    start = time.time()
