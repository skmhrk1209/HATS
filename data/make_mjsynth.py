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
def make_mjsynth(filenames, directory, image_size, thread_id):

    for i, filename in tqdm(filenames):

        string = os.path.splitext(os.path.basename(filename))[0].split("_")[1]
        image = cv2.imread(filename)

        h = image.shape[0]
        w = image.shape[1]
        y = random.randint(0, image_size[0] - h)
        x = random.randint(0, image_size[1] - w)

        image = np.pad(
            array=image,
            pad_width=[
                [y, image_size[0] - h - y],
                [x, image_size[1] - w - x]
            ],
            mode="constant",
            constant_values=0
        )

        cv2.imwrite(os.path.join(directory, "{}_{}.jpg".format(i, string)), image)


if __name__ == "__main__":

    filenames = [
        filename for filename in tqdm(glob.glob("/home/sakuma/data/mnt/*/*/*/*/*/*.jpg"))
        if ((lambda string: len(string) <= 10)(os.path.splitext(os.path.basename(filename))[0].split("_")[1]) and
            (lambda image: image is not None and all([l1 <= l2 for l1, l2 in zip(image.shape[:2], [256, 256])]))(cv2.imread(filename)))
    ]

    random.seed(0)
    random.shuffle(filenames)

    make_multi_thread(make_mjsynth, num_threads=32, split=True)(
        list(enumerate(filenames[:900000])),
        "/home/sakuma/data/mjsynth/train"
    )

    make_multi_thread(make_mjsynth, num_threads=32, split=True)(
        list(enumerate(filenames[900000:1000000])),
        "/home/sakuma/data/mjsynth/test",
    )
