import os
import tensorflow as tf

dir = "../data/synth90k/"

with tf.python_io.TFRecordWriter("synth90k_train.tfrecord") as writer:
    class_ids = {}
    class_ids.update({chr(j): i for i, j in enumerate(range(ord("0"), ord("9") + 1), 0)})
    class_ids.update({chr(j): i for i, j in enumerate(range(ord("A"), ord("Z") + 1), class_ids["9"] + 1)})
    class_ids.update({"": max(class_ids.values()) + 1})
    with open(os.path.join(dir, "annotation_train.txt")) as f:
        for line in f:
            path = os.path.join(dir, line.split()[0])
            label = os.path.splitext(os.path.basename(path))[0].split("_")[1]
            label = label.upper()
            label = list(label)
            label = map(lambda char: class_ids[char], label)
            label = np.pad(
                array=list(label),
                pad_width=[[0, 23 - len(label)]],
                mode="constant",
                constant_values=class_ids[""]
            )

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
