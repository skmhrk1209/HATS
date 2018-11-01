import numpy as np
import cv2
import glob
import os
import shutil
from shapely.geometry import box

filenames = glob.glob("/home/sakuma/data/mjsynth/train/*.jpg")

for i in range(90000):

    num_strings = np.random.randint(1, 5)
    random_filenames = np.random.choice(filenames, num_strings)
    image = np.zeros([256, 256, 3], dtype=np.uint8)

    boxes = []

    for random_filename in random_filenames:

        random_image = cv2.imread(random_filename)

        while True:

            h = random_image.shape[0]
            w = random_image.shape[1]
            x = np.random.randint(0, 256 - w)
            y = np.random.randint(0, 256 - h)
            proposal = box(x, y, x + w, y + h)

            for box in boxes:

                if proposal.intersects(box):
                    break

            else:

                image[y:y+h, x:x+w, :] += random_image
                boxes.append(proposal)

    random_filenames = [pair[0] for pair in sorted(zip(random_filenames, boxes), key=lambda pair: pair[1])]
    labels = "_".join([os.path.splitext(random_filename)[0].split("_")[1] for random_filename in random_filenames])

    cv2.imwrite("/home/sakuma/data/multi_mjsynth/train/{}_{}.jpg".format(i, labels), image)

filenames = glob.glob("/home/sakuma/data/mjsynth/test/*.jpg")

for i in range(10000):

    num_strings = np.random.randint(1, 5)
    random_filenames = np.random.choice(filenames, num_strings)
    image = np.zeros([256, 256, 3], dtype=np.uint8)

    boxes = []

    for random_filename in random_filenames:

        random_image = cv2.imread(random_filename)

        while True:

            h = random_image.shape[0]
            w = random_image.shape[1]
            x = np.random.randint(0, 256 - w)
            y = np.random.randint(0, 256 - h)
            proposal = box(x, y, x + w, y + h)

            for box in boxes:

                if proposal.intersects(box):
                    break

            else:

                image[y:y+h, x:x+w, :] += random_image
                boxes.append(proposal)

    random_filenames = [pair[0] for pair in sorted(zip(random_filenames, boxes), key=lambda pair: pair[1])]
    labels = "_".join([os.path.splitext(random_filename)[0].split("_")[1] for random_filename in random_filenames])

    cv2.imwrite("/home/sakuma/data/multi_mjsynth/test/{}_{}.jpg".format(i, labels), image)
