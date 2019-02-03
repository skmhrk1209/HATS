import tensorflow as tf
import cv2
import sys

class_ids = {}
class_ids.update({chr(j): i for i, j in enumerate(range(ord("0"), ord("9") + 1), 0)})
class_ids.update({chr(j): i for i, j in enumerate(range(ord("A"), ord("Z") + 1), class_ids["9"] + 1)})
class_ids.update({"": max(class_ids.values()) + 1})
class_names = {class_id: class_name for class_name, class_id in class_ids}

for record in tf.python_io.tf_record_iterator(sys.argv[1]):

    example = tf.train.Example()
    example.ParseFromString(record)

    image = cv2.imread(example.features.feature["path"].bytes_list.value[0].decode())
    label = "".join([class_names[class_id] for class_id in example.features.feature["label"].int64_list.value])

    print(label)
    cv2.imshow("image", image)

    if cv2.waitKey() == ord("q"):
        break
