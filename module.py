import tensorflow as tf
import tensorflow_hub as hub
import numpy as np

module = hub.Module("https://tfhub.dev/google/imagenet/resnet_v2_50/feature_vector/1")
height, width = hub.get_expected_image_size(module)
images = np.zeros([1, 128, 128, 3])
features = module(images)  # Features with shape [batch_size, num_features].

with tf.Session() as sess:

    sess.run(tf.global_variables_initializer())

    print(sess.run(features))