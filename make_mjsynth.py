import numpy as np
import cv2
import os
import glob
import shutil
import random
import threading
import itertools
from numba import jit
from tqdm import tqdm
from shapely.geometry import box


def make_multi_thread(func, num_threads):

    def func_mt(*args, **kwargs):

        size = len(args[0]) // num_threads

        threads = [threading.Thread(
            target=func,
            args=(args[0][size * i: size * (i + 1) if i < num_threads - 1 else len(args[0]) - 1],) + args[1:],
            kwargs=kwargs
        ) for i in range(num_threads)]

        for thread in threads:
            thread.start()

        for thread in threads:
            thread.join()

    return func_mt


@jit(nopython=False, nogil=True)
def make_mjsynth(filenames, directory):

    for i, filename in tqdm(filenames):

        shutil.move(filename, os.path.join(directory, "{}_{}.jpg".format(i, os.path.splitext(os.path.basename(filename))[0].split("_")[1])))


if __name__ == "__main__":

    filenames = [
        filename for filename in glob.glob("/home/sakuma/data/synth/*/*/*.jpg")
        if len(os.path.splitext(os.path.basename(filename))[0].split("_")[1]) <= 10
    ]

    make_multi_thread(make_mjsynth, num_threads=32)(
        list(enumerate(filenames[:int(len(filenames) * 0.9)])),
        "/home/sakuma/data/multi_mjsynth/train"
    )

    make_multi_thread(make_mjsynth, num_threads=32)(
        list(enumerate(filenames[int(len(filenames) * 0.9):])),
        "/home/sakuma/data/multi_mjsynth/test"
    )
