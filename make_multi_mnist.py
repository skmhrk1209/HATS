import tensorflow as tf

mnist = tf.keras.datasets.mnist
(train_images, train_labels),(test_images, y_test) = mnist.load_data()
train_images, test_images = train_images / 255.0, test_images / 255.0
