import numpy as np
import cv2
import glob
import os
import shutil

filenames = glob.glob("~/data/synth/*/*/*.jpg")

filenames = [filename for filename in filenames if (lambda image: (image is not None) and (image.shape[1] <= 128))(cv2.imread(filename))]

for i, filename in enumerate(filenames[:int(len(filenames) * 0.9)]):

    shutil.move(filename, "~/data/mjsynth/train/{}_{}.jpg".format(i, filename.split("_")[1]))

for i, filename in enumerate(filenames[int(len(filenames) * 0.9):]):

    shutil.move(filename, "~/data/mjsynth/test/{}_{}.jpg".format(i, filename.split("_")[1]))

filenames = glob.glob("~/data/mjsynth/train/*.jpg")
print(len(filenames))

for i in range(900000):

    random_filenames = np.random.choice(filenames, np.random.randint(1, 5))
    labels = "_".join([os.path.splitext(random_filename)[0].split("_")[1] for random_filename in random_filenames])
    random_filenames = np.pad(random_filenames, [0, 4 - len(random_filenames)], "constant", constant_values="")
    np.random.shuffle(random_filenames)
    image = np.zeros([256, 256, 3], dtype=np.uint8)

    for i, random_filename in enumerate(random_filenames):

        if not random_filename:
            continue

        random_image = cv2.imread(random_filename)

        x = np.random.randint(0, 256 - random_image.shape[1])
        y = np.random.randint(64 * i, 64 * (i + 1) - random_image.shape[0])

        image[y: y + random_image.shape[0], x: x + random_image.shape[1], :] = random_image

    cv2.imwrite("~/data/multi_mjsynth/train/{}_{}.jpg".format(i, labels), image)

filenames = glob.glob("~/data/mjsynth/test/*.jpg")
print(len(filenames))

for i in range(100000):

    random_filenames = np.random.choice(filenames, np.random.randint(1, 5))
    random_filenames = np.pad(random_filenames, [0, 4 - len(random_filenames)], "constant", constant_values="")
    np.random.shuffle(random_filenames)
    labels = "_".join([os.path.splitext(random_filename)[0].split("_")[1] for random_filename in random_filenames if random_filename])
    image = np.zeros([256, 256, 3], dtype=np.uint8)

    for i, random_filename in enumerate(random_filenames):

        if not random_filename:
            continue

        random_image = cv2.imread(random_filename)

        x = np.random.randint(0, 256 - random_image.shape[1])
        y = np.random.randint(64 * i, 64 * (i + 1) - random_image.shape[0])

        image[y: y + random_image.shape[0], x: x + random_image.shape[1], :] = random_image

    cv2.imwrite("~/data/multi_mjsynth/test/{}_{}.jpg".format(i, labels), image)
