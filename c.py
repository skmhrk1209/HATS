import os
import cv2
import tensorflow as tf
import numpy as np
from tqdm import tqdm

dir = "../data/synth90k/"

with open(os.path.join(dir, "annotation_test.txt")) as f:
    for line in tqdm(f):
        path = os.path.join(dir, line.split()[0])
        if cv2.imread(path) is None:
            print(path)