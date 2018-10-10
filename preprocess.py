import tensorflow as tf
import numpy as np
import glob
import cv2

resnet50 = tf.keras.applications.resnet50.ResNet50(weights="imagenet", include_top=False)

filenames = glob.glob("/home/sakuma/data/imagenet/*/*/*")

for filename in filenames:

    image = tf.keras.preprocessing.image.load_img(filename, target_size=(224, 224))
    image = tf.keras.preprocessing.image.img_to_array(image)
    image = np.expand_dims(image, axis=0)
    image = tf.keras.applications.resnet50.preprocess_input(image)
    tf.keras.preprocessing.image.save_img(filename.replace("imagenet", "preprocessed_imagenet"), image[0])
    print("{} processed".format(filename))
