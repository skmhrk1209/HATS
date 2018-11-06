import numpy as np
import cv2
import glob
import os
import random
import threading
import itertools
from numba import jit
from tqdm import trange
from shapely.geometry import box


def make_multi_thread(func, num_threads):

    def func_mt(*args):

        threads = [threading.Thread(target=func, args=args + (i,)) for i in range(num_threads)]

        for thread in threads:
            thread.start()

        for thread in threads:
            thread.join()

    return func_mt


@jit(nopython=False, nogil=True)
def make_multi_mjsynth(filenames, num_data, image_size, sequence_length, string_length, thread_id):

    for i in range(num_data * thread_id, num_data * (thread_id + 1)):

        num_strings = random.randint(1, sequence_length)
        random_filenames = random.sample(filenames, num_strings)
        image = np.zeros(image_size + [3], dtype=np.uint8)

        random_rects = []

        for random_filename in random_filenames[:]:

            random_image = cv2.imread(random_filename)

            if random_image.shape[0] > image_size[0] or random_image.shape[1] > image_size[1]:
                random_filenames.remove(random_filename)
                continue

            for j in itertools.count():

                h = random_image.shape[0]
                w = random_image.shape[1]
                y = random.randint(0, image_size[0] - h)
                x = random.randint(0, image_size[1] - w)
                proposal = (y, x, y + h, x + w)

                for random_rect in random_rects:

                    if box(*proposal).intersects(box(*random_rect)):
                        break

                else:

                    image[y:y+h, x:x+w, :] += random_image
                    random_rects.append(proposal)
                    break

            print j

        random_filenames = [random_filename for random_rect, random_filename in sorted(zip(random_rects, random_filenames))]
        labels = "_".join([os.path.splitext(os.path.basename(random_filename))[0].split("_")[1] for random_filename in random_filenames])

        cv2.imwrite(os.path.join(os.path.dirname(filenames[0]).replace("mjsynth", "multi_mjsynth"), "{}_{}.jpg".format(i, labels)), image)


if __name__ == "__main__":

    make_multi_thread(make_multi_mjsynth, 32)(glob.glob("/home/sakuma/data/mjsynth/train/*"), 3000, [256, 256], 4, 10)
    make_multi_thread(make_multi_mjsynth, 32)(glob.glob("/home/sakuma/data/mjsynth/test/*"), 300, [256, 256], 4, 10)
