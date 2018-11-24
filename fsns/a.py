import tensorflow as tf
import cv2
import numpy as np

for record  in tf.python_io.tf_record_iterator("train.tfrecord"):
    example = tf.train.Example()
    example.ParseFromString(record)
 
    path = example.features.feature["path"].bytes_list.value[0].decode("utf-8")
    image = cv2.imread(path)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    label = example.features.feature["label"].int64_list.value
 
    print(path)
    print(label)
    cv2.imshow("", image)
    cv2.waitKey()