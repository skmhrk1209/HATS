import numpy as np
import cv2
import os
import glob
import shutil
import random
from tqdm import tqdm

if __name__ == "__main__":

    filenames = [
        filename for filename in glob.glob("/home/sakuma/data/synth/*/*/*.jpg")
        if len(os.path.splitext(os.path.basename(filename))[0].split("_")[1]) <= 10
    ]

    filenames = random.sample(filenames, 96000 + 9600)

    for i, filename in enumerate(tqdm(filenames[:96000])):

        shutil.move(filename, os.path.join("/home/sakuma/data/mjsynth/train",
                                           "{}_{}.jpg".format(i, os.path.splitext(os.path.basename(filename))[0].split("_")[1])))

    for i, filename in enumerate(tqdm(filenames[96000:])):

        shutil.move(filename, os.path.join("/home/sakuma/data/mjsynth/test",
                                           "{}_{}.jpg".format(i, os.path.splitext(os.path.basename(filename))[0].split("_")[1])))
