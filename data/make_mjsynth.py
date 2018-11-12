import numpy as np
import cv2
import os
import glob
import shutil
import random
import threading
import itertools
from numba import jit
from tqdm import tqdm
from shapely.geometry import box

if __name__ == "__main__":

    filenames = [
        filename for filename in glob.glob("/home/sakuma/data/mnt/*/*/*/*/*/*.jpg")
        if len(os.path.splitext(os.path.basename(filename))[0].split("_")[1]) <= 10
    ]

    random.seed(0)
    random.shuffle(filenames)

    for i, filename in enumerate(tqdm(filenames[:int(len(filenames) * 0.9)])):

        shutil.copy(filename, os.path.join("/home/sakuma/data/mjsynth/train",
                                           "{}_{}.jpg".format(i, os.path.splitext(os.path.basename(filename))[0].split("_")[1])))

    for i, filename in enumerate(tqdm(filenames[int(len(filenames) * 0.9):])):

        shutil.copy(filename, os.path.join("/home/sakuma/data/mjsynth/test",
                                           "{}_{}.jpg".format(i, os.path.splitext(os.path.basename(filename))[0].split("_")[1])))
