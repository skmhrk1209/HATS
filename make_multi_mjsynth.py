import numpy as np
import cv2
import glob
import os
import shutil
import threading
from operator import itemgetter
from operator import attrgetter
from numba import jit
from shapely.geometry import box


def make_multi_thread(func, num_threads):

    def func_mt(*args):

        chunk_size = len(args[0]) / num_threads
        chunks = [args[0][chunk_size * i:chunk_size * (i + 1)] for i in range(num_threads)[:-1]]
        chunks += [args[0][chunk_size * i:] for i in range(num_threads)[-1:]]

        threads = [threading.Thread(target=func, args=(chunk,) + args[1:]) for chunk in chunks]

        for thread in threads:
            thread.start()

        for thread in threads:
            thread.join()

    return func_mt


@jit(nopython=True, nogil=True)
def make_multi_mjsynth(filenames, sequence_length, image_size, num_data):

    for i in range(num_data):

        num_strings = np.random.random_integers(1, sequence_length)
        random_filenames = np.random.choice(filenames, num_strings)
        image = np.zeros(image_size + [3], dtype=np.uint8)

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

        random_filenames = np.delete(random_filenames, remove_indices)
        random_filenames = [random_filename for random_rect, random_filename in sorted(zip(random_rects, random_filenames))]
        labels = "_".join([os.path.splitext(os.path.basename(random_filename))[0].split("_")[1] for random_filename in random_filenames])
        print(i, labels)

        cv2.imwrite(os.path.join(os.path.dirname(filenames[0]).replace("mjsynth", "multi_mjsynth"), "{}_{}.jpg".format(i, labels)), image)


if __name__ == "__main__":

    make_multi_thread(make_multi_mjsynth, 16)(glob.glob("/home/sakuma/data/mjsynth/train/*"), 4, [256, 256], 90000)
    make_multi_thread(make_multi_mjsynth, 16)(glob.glob("/home/sakuma/data/mjsynth/test/*"), 4, [256, 256], 10000)
