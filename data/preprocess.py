import tensorflow as tf
import numpy as np
import argparse
import glob
import os
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications import resnet50

parser = argparse.ArgumentParser()
parser.add_argument("--imagenet_dir", type=str)
args = parser.parse_args()

filenames = glob.glob(os.path.join(args.imagenet_dir, "/*/*/*.JPEG"))

for filename in filenames:

    img = image.load_img(filename, target_size=(224, 224))
    array = image.img_to_array(img)
    array = resnet50.preprocess_input(array)
    print(type(array))