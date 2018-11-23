import tensorflow as tf
import glob
import cv2
import numpy as np

for filename in [filename for filename in glob.glob("/home/sakuma/data/fsns/*") if "train" in filename]:

    for record  in tf.python_io.tf_record_iterator(filename):

        example = tf.train.Example()
        example.ParseFromString(record)
    
        image = example.features.feature["image/encoded"].bytes_list.value[0]
        label = example.features.feature["image/text"].bytes_list.value[0]
    
        image = np.fromstring(image, dtype=np.uint8)
        image = image.reshape([150, 600])
        
        cv2.imshow("", image)
        cv2.waitKey()