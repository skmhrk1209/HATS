import numpy as np
import cv2
import glob
import os
import shutil
from operator import itemgetter
from operator import attrgetter
from numba import jit
from shapely.geometry import box


@jit
def prepare_mjsynth(mjsynth_dir, string_length, split_ratio):

    filenames = glob.glob(os.path.join(mjsynth_dir, "*"))

    for i, filename in enumerate(filenames):

        label = os.path.splitext(os.path.basename(filename))[0].split("_")[1]
        print(i, label)

        if len(label) <= string_length:

            if i < len(filenames) * split_ratio:
                shutil.move(filename, "/home/sakuma/mjsynth/train/{}_{}.jpg".format(i, label))
            else:
                shutil.move(filename, "/home/sakuma/mjsynth/test/{}_{}.jpg".format(i, label))


@jit
def make_multi_mjsynth(mjsynth_dir, image_size, num_data, sequence_length):

    filenames = glob.glob(os.path.join(mjsynth_dir, "*"))

    for i in range(num_data):

        num_strings = np.random.random_integers(1, sequence_length)
        random_filenames = np.random.choice(filenames, num_strings)
        image = np.zeros(image_size + (3,), dtype=np.uint8)

        random_rects = []
        remove_indices = []

        for j, random_filename in enumerate(random_filenames):

            random_image = cv2.imread(random_filename)

            if random_image.shape[0] > image_size[0] or random_image.shape[1] > image_size[1]:
                remove_indices.append(j)
                continue

            while True:

                h = random_image.shape[0]
                w = random_image.shape[1]
                y = np.random.random_integers(0, image_size[0] - h)
                x = np.random.random_integers(0, image_size[1] - w)
                proposal = (y, x, y + h, x + w)

                for random_rect in random_rects:

                    if box(*proposal).intersects(box(*random_rect)):
                        break

                else:

                    image[y:y+h, x:x+w, :] += random_image
                    random_rects.append(proposal)
                    break
        '''
        random_filenames = np.delete(random_filenames, remove_indices)
        random_filenames = [random_filename for random_rect, random_filename in sorted(zip(random_rects, random_filenames))]
        labels = "_".join([os.path.splitext(os.path.basename(random_filename))[0].split("_")[1] for random_filename in random_filenames])
        print(i, labels)
        '''
        cv2.imwrite(os.path.join(mjsynth_dir.replace("mjsynth", "multi_mjsynth"), "{}_{}.jpg".format(i, labels)), image)


if __name__ == "__main__":

    prepare_mjsynth("/home/sakuma/synth/*/*", 10, 0.9)
    make_multi_mjsynth("/home/sakuma/mjsynth/train", (256, 256), 90000, 4)
    make_multi_mjsynth("/home/sakuma/mjsynth/test", (256, 256), 10000, 4)
