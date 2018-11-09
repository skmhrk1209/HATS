import tensorflow as tf


class Dataset(object):

    def __init__(self, filenames, num_epochs, batch_size, buffer_size):

        self.dataset = tf.data.TFRecordDataset(filenames)
        self.dataset = self.dataset.shuffle(buffer_size)
        self.dataset = self.dataset.repeat(num_epochs)
        self.dataset = self.dataset.map(self.parse)
        self.dataset = self.dataset.batch(batch_size)
        self.dataset = self.dataset.prefetch(1)
        self.iterator = self.dataset.make_one_shot_iterator()

    def parse(self, example):

        raise NotImplementedError()

    def get_next(self):

        return self.iterator.get_next()
