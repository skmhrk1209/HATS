import os
import sys
import tensorflow as tf
import numpy as np
import skimage
from tqdm import *

class_ids = {}
class_ids.update({chr(j): i for i, j in enumerate(range(ord("0"), ord("9") + 1), 0)})
class_ids.update({chr(j): i for i, j in enumerate(range(ord("A"), ord("Z") + 1), class_ids["9"] + 1)})
class_ids.update({"": max(class_ids.values()) + 1})

with tf.python_io.TFRecordWriter(sys.argv[2]) as writer:

    with open(sys.argv[1]) as f:

        for line in tqdm(f):

            path = os.path.join(os.path.dirname(sys.argv[1]), line.split()[0])

            try:
                skimage.io.imread(path)
            except Exception as error:
                print(path)
                print(error)
                continue

            label = os.path.splitext(os.path.basename(path))[0].split("_")[1]
            label = list(map(lambda char: class_ids[char], label.upper()))
            label = np.pad(label, [[0, 23 - len(label)]], "constant", constant_values=class_ids[""])

            writer.write(
                record=tf.train.Example(
                    features=tf.train.Features(
                        feature={
                            "path": tf.train.Feature(
                                bytes_list=tf.train.BytesList(
                                    value=[path.encode("utf-8")]
                                )
                            ),
                            "label": tf.train.Feature(
                                int64_list=tf.train.Int64List(
                                    value=label.tolist()
                                )
                            )
                        }
                    )
                ).SerializeToString()
            )
