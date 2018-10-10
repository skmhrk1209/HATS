import tensorflow as tf
import numpy as np
import argparse
import glob
import os

parser = argparse.ArgumentParser()
parser.add_argument("--imagenet_dir", type=str, default="../data/imagenet", help="model directory")
args = parser.parse_args()

resnet50 = tf.keras.applications.resnet50.ResNet50(weights="imagenet", include_top=False)

filenames = glob.glob(os.path.join(args.imagenet_dir, "/*/*/*.JPEG"))
print(len(filenames))

for filename in filenames:

    image = tf.keras.preprocessing.image.load_img(filename, target_size=(224, 224))
    image = tf.keras.preprocessing.image.img_to_array(image)
    image = np.expand_dims(image, axis=0)
    image = tf.keras.applications.resnet50.preprocess_input(image)
    np.save(filename.replace("imagenet", "imagenet_preprocessed").replace(".JPEG", ".npy"), image)
    print("{} processed".format(filename))
