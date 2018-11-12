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

    def func_mt(arg, *args, **kwargs):

        chunk_size = len(arg) // num_threads
        chunk_args = [arg[chunk_size * i: chunk_size * (i + 1)] for i in range(num_threads)[:-1]]
        chunk_args += [arg[chunk_size * i:] for i in range(num_threads)[-1:]]

        threads = [threading.Thread(
            target=func,
            args=[chunk_arg] + args,
            kwargs=dict(kwargs, thread_id=i)
        ) for chunk_arg in range(chunk_args)]

        for thread in threads:
            thread.start()

        for thread in threads:
            thread.join()

    return func_mt


@jit(nopython=False, nogil=True)
def make_mjsynth(filenames, directory, string_length, thread_id):

    for i, filename in tqdm(filenames):

        string = os.path.splitext(os.path.basename(filename))[0].split("_")[1]

        if len(string) >= string_length:
            continue

        shutil.copy(filename, os.path.join(directory, "{}_{}.jpg".format(i, string)))


if __name__ == "__main__":

    filenames = glob.glob("/home/sakuma/data/synth/*/*/*.jpg")

    make_multi_thread(make_mjsynth, num_threads=32)(
        filenames[:int(len(filenames) * 0.9)],
        directory="/home/sakuma/data/mjsynth/train",
        string_length=10
    )

    make_multi_thread(make_mjsynth, num_threads=32)(
        filenames[int(len(filenames) * 0.9):],
        directory="/home/sakuma/data/mjsynth/test",
        string_length=10
    )
