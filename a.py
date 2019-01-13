import tensorflow as tf
import numpy as np
import glob
import sys
import os
import re
import cv2


def main(input_directory, output_filename, sequence_length):

    with open(os.path.join(input_directory, "gt.txt"), encoding="utf-8-sig") as f:

        regex = re.compile(r'(.+), "(.+)"')

        for line in f:

            filename, label = regex.findall(line.strip())[0]
            image = cv2.imread(os.path.join(input_directory, filename))
            cv2.imwrite(os.path.join(input_directory, "{}.jpg".format(label)}), image)


if __name__ == "__main__":

    main(*sys.argv[1:3], int(sys.argv[3]))
