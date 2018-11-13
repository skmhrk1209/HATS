import numpy as np
import cv2
import os
import glob
import shutil
import random
import threading
import itertools
from numba import jit
from tqdm import tqdm, trange
from shapely.geometry import box


def make_multi_thread(func, num_threads, split=False):

    def func_mt(*args, **kwargs):

        if split:

            threads = [
                threading.Thread(
                    target=func,
                    args=(arg,) + args[1:],
                    kwargs=dict(kwargs, thread_id=i)
                ) for i, arg in enumerate(np.array_split(args[0], num_threads))
            ]

        else:

            threads = [
                threading.Thread(
                    target=func,
                    args=args,
                    kwargs=dict(kwargs, thread_id=i)
                ) for i in range(num_threads)
            ]

        for thread in threads:
            thread.start()

        for thread in threads:
            thread.join()

    return func_mt


@jit(nopython=False, nogil=True)
def make_multi_mjsynth(filenames, directory, num_data, image_size, sequence_length, num_retries, thread_id):

    for i in trange(num_data * thread_id, num_data * (thread_id + 1)):

        multi_image = np.zeros(image_size + (3,), dtype=np.uint8)

        strings = []
        rects = []

        for filename in random.sample(filenames, random.randint(1, sequence_length)):

            string = os.path.splitext(os.path.basename(filename))[0].split("_")[1]
            image = cv2.imread(filename)

            for _ in range(num_retries):

                h = image.shape[0]
                w = image.shape[1]
                y = random.randint(0, image_size[0] - h)
                x = random.randint(0, image_size[1] - w)
                proposal = (y, x, y + h, x + w)

                for rect in rects:

                    if box(*proposal).intersects(box(*rect)):
                        break

                else:

                    multi_image[y:y+h, x:x+w, :] += image
                    strings.append(string)
                    rects.append(proposal)
                    break

        strings = [string for rect, string in sorted(zip(rects, strings))]
        cv2.imwrite(os.path.join(directory, "{}_{}.jpg".format(i, "_".join(strings))), multi_image)


if __name__ == "__main__":

    filenames = [
        filename for filename in tqdm(glob.glob("/home/sakuma/data/mnt/*/*/*/*/*/*.jpg"))
        if ((lambda string: len(string) <= 10)(os.path.splitext(os.path.basename(filename))[0].split("_")[1]) and
            (lambda image: image is not None and all([l1 <= l2 for l1, l2 in zip(image.shape[:2], [256, 256])]))(cv2.imread(filename)))
    ]

    random.seed(0)
    random.shuffle(filenames)

    make_multi_thread(make_multi_mjsynth, num_threads=32, split=False)(
        filenames[:int(len(filenames) * 0.9)],
        "/home/sakuma/data/multi_mjsynth/train",
        num_data=28125,
        image_size=(256, 256),
        sequence_length=4,
        num_retries=100
    )

    make_multi_thread(make_multi_mjsynth, num_threads=32, split=False)(
        filenames[int(len(filenames) * 0.9):],
        "/home/sakuma/data/multi_mjsynth/test",
        num_data=3125,
        image_size=(256, 256),
        sequence_length=4,
        num_retries=100
    )
