import numpy as np
import cv2
import glob
import os
import shutil
from shapely.geometry import box

filenames = glob.glob("/home/sakuma/data/synth/*/*/*.jpg")

print(len(filenames))

i = 0
for filename in filenames:

    label = os.path.splitext(os.path.basename(filename)).split("_")[1]

    if len(label) <= 10:

        shutil.move(filename, "/home/sakuma/data/mjsynth/{}_{}.jpg".format(i, label))
        i += 1

    