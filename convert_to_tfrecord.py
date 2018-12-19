import numpy as np
import scipy
import glob
import sys
import os


def main(input_directory, output_filename, sequence_lengths):

    filenames = glob.glob(os.path.join(input_directory, "*"))
    datasets = [scipy.io.loadmat(filename) for filename in filenames]

    with tf.python_io.TFRecordWriter(output_filename) as writer:

        for filename, dataset in zip(filenames, datasets):

            label = np.concatenate(dataset["rectgt"][:, -2])
            label = map_innermost_element(list, label)
            label = map_innermost_element(lambda char: ord(char) - 32, label)

            for i, sequence_length in enumerate(sequence_lengths[::-1]):

                label = map_innermost_list(
                    function=lambda sequence: np.pad(
                        array=sequence,
                        pad_width=[[0, sequence_length - len(sequence)]] + [[0, 0]] * i,
                        mode="constant",
                        constant_values=95
                    ),
                    sequence=label
                )

            writer.write(
                record=tf.train.Example(
                    features=tf.train.Features(
                        feature={
                            "path": tf.train.Feature(
                                bytes_list=tf.train.BytesList(
                                    value=[filename.replace("images", "labels").replace("mat", "jpg").encode("utf-8")]
                                )
                            ),
                            "label": tf.train.Feature(
                                int64_list=tf.train.Int64List(
                                    value=label.astype(np.int32).reshape([-1]).tolist()
                                )
                            )
                        }
                    )
                ).SerializeToString()
            )


if __name__ == "__main__":

    main(*sys.argv[1:3], sys.argv[3:])
