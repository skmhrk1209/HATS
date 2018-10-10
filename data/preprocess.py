import tensorflow as tf
import numpy as np
import argparse
import glob
import os

parser = argparse.ArgumentParser()
parser.add_argument("--imagenet_dir", type=str)
args = parser.parse_args()

filenames = glob.glob("/home/sakuma/data/imagenet/*/*/*.JPEG")

for filename in filenames:

    image = tf.keras.preprocessing.image.load_img(filename, target_size=(224, 224))
    image = tf.keras.preprocessing.image.img_to_array(image)
    image = tf.keras.applications.resnet50.preprocess_input(image)
    np.save(filename.replace("imagenet", "imagenet_preprocessed").replace("JPEG", "npy"))