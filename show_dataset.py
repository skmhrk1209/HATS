from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import argparse
import cv2

parser = argparse.ArgumentParser()
parser.add_argument("filename", type=str)
args = parser.parse_args()

for record in tf.python_io.tf_record_iterator(args.filename):

    example = tf.train.Example()
    example.ParseFromString(record)

    string = example.features.feature["image"].bytes_list.value[0]
    image = np.fromstring(string, dtype=np.uint8).reshape([128, 128, 3])

    cv2.imshow("image", image)

    if cv2.waitKey(1000) == ord("q"):

        break
