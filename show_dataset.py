import tensorflow as tf
import sys
import cv2

for record in tf.python_io.tf_record_iterator(sys.argv[1]):

    example = tf.train.Example()
    example.ParseFromString(record)
 
    path = example.features.feature["path"].bytes_list.value[0].decode("utf-8")
    length = example.features.feature["length"].int64_list.value[0]
    label = example.features.feature["length"].int64_list.value[:5]
    top = example.features.feature["top"].int64_list.value[0]
    bottom = example.features.feature["bottom"].int64_list.value[0]
    left = example.features.feature["left"].int64_list.value[0]
    right = example.features.feature["right"].int64_list.value[0]

    image = cv2.imread("data/test/" + path)
    image = cv2.rectangle(image, (left, top), (right, bottom), (255, 0, 0))
 
    cv2.imshow("", image)

    print(length)

    if cv2.waitKey(1000) == ord("q"): break