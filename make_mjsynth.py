import numpy as np
import cv2
import os
import glob
import shutil
from tqdm import tqdm

if __name__ == "__main__":

    filenames = [
        filename for filename in glob.glob("/home/sakuma/data/synth/*/*/*.jpg")
        if len(os.path.splitext(os.path.basename(filename))[0].split("_")[1]) <= 10
    ]

    for i, filename in enumerate(tqdm(filenames[:int(len(filenames) * 0.9)])):

        shutil.move(filename, os.path.join("/home/sakuma/data/mjsynth/train",
                                           "{}_{}.jpg".format(i, os.path.splitext(os.path.basename(filename))[0].split("_")[1])))

    for i, filename in enumerate(tqdm(filenames[int(len(filenames) * 0.9):])):

        shutil.move(filename, os.path.join("/home/sakuma/data/mjsynth/test",
                                           "{}_{}.jpg".format(i, os.path.splitext(os.path.basename(filename))[0].split("_")[1])))
