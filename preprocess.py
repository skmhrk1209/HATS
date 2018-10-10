import tensorflow as tf
import numpy as np
import glob
import cv2

resnet50 = tf.keras.applications.resnet50.ResNet50(weights="imagenet", include_top=False)

filenames = glob.glob("~/data/imagenet/*/*/*")

for filename in filenames:

    image = cv2.imread(filename)
    image = cv2.resize(image, (224, 224))
    image = np.expand_dims(image, axis=0)
    image = tf.keras.applications.resnet50.preprocess_input(image)

    cv2.imwrite(filename.replace("imagenet", "preprocessed_imagenet"), image[0])
    print("{} processed".format(filename))