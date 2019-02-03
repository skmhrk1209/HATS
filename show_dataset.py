import tensorflow as tf
import cv2
import sys

for record in tf.python_io.tf_record_iterator(sys.argv[1]):

    example = tf.train.Example()
    example.ParseFromString(record)

    path = example.features.feature["path"].bytes_list.value[0].decode()
    label = example.features.feature["label"].bytes_list.value[0].decode()

    print(label)
    cv2.imshow("image", cv2.imread(path))

    if cv2.waitKey() == ord("q"):
        break
