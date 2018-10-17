import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
from tensorflow.python.tools import freeze_graph

module = hub.Module("https://tfhub.dev/google/imagenet/resnet_v2_50/feature_vector/1")
height, width = hub.get_expected_image_size(module)
images = np.zeros([1, 128, 128, 3])
features = module(images)

with tf.Session() as sess:

    sess.run(tf.global_variables_initializer())
    print(sess.run(features))

    checkpoint = tf.train.Saver().save(
        sess=sess,
        save_path="resnet_v2_50/model.ckpt"
    )

    tf.train.write_graph(
        graph_or_graph_def=sess.graph.as_graph_def(),
        logdir="resnet_v2_50",
        name="graph.pb",
        as_text=False
    )

    freeze_graph.freeze_graph(
        input_graph="resnet_v2_50/graph.pbtxt",
        input_saver="",
        input_binary=False,
        input_checkpoint="resnet_v2_50/model.ckpt",
        output_node_names="fakes",
        restore_op_name="",
        filename_tensor_name="",
        output_graph="resnet_v2_50/frozen_graph.pb",
        clear_devices=True,
        initializer_nodes=""
    )
