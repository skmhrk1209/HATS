import tensorflow as tf
import numpy as np
import glob
import sys
import os
import re
import cv2


def main(input_directory):

    with open(os.path.join(input_directory, "gt.txt"), encoding="utf-8-sig") as f:

        for line in f:

            filename, label = line.strip().split(",")
            label = label.strip('"')
            image = cv2.imread(os.path.join(input_directory, filename))
            cv2.imwrite(os.path.join(input_directory, "{}.jpg".format(label)), image)


if __name__ == "__main__":

    main(sys.argv[1])
