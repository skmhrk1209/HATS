import tensorflow as tf
import argparse
import os
import glob
import cv2

parser = argparse.ArgumentParser()
parser.add_argument("--filename", type=str, required=True, help="tfrecord filename")
args = parser.parse_args()

for record  in tf.python_io.tf_record_iterator(args.filename):
    example = tf.train.Example()
    example.ParseFromString(record)

    filename = example.features.feature["path"].bytes_list.value[0]
    image = cv2.imread(filename)
 
    cv2.imshow("", image)
    cv2.waitKey(1000)