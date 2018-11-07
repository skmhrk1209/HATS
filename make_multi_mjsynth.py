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


def make_multi_thread(func, num_threads, split):

    def func_mt(arg, *args, **kwargs):

        def merge(*ds):
            md = {}
            for d in ds:
                md.update(d)
            return md

        if split:

            chunk_len = len(arg) / num_threads
            chunk_args = [arg[chunk_len * i: chunk_len * (i + 1)] for i in range(num_threads)[:-1]]
            chunk_args += [arg[chunk_len * i:] for i in range(num_threads)[-1:]]

            threads = [threading.Thread(
                target=func,
                args=(chunk_arg,) + args,
                kwargs=merge(kwargs, {"thread_id": i})
            ) for i, chunk_arg in enumerate(chunk_args)]

        else:

            threads = [threading.Thread(
                target=func,
                args=(arg,) + args,
                kwargs=merge(kwargs, {"thread_id": i})
            ) for i in range(num_threads)]

        for thread in threads:
            thread.start()

        for thread in threads:
            thread.join()

    return func_mt


@jit(nopython=False, nogil=True)
def prepare_mjsynth(filenames, directory, image_size, string_length, thread_id):

    os.mkdir(os.path.join(directory, "{}".format(thread_id)))

    for i, filename in enumerate(tqdm(filenames)):

        string = os.path.splitext(os.path.basename(filename))[0].split("_")[1]

        if len(string) <= string_length:

            image = cv2.imread(filename)

            if image is not None and image.shape[0] <= image_size[0] and image.shape[1] <= image_size[1]:

                shutil.copy(filename, os.path.join(directory, "{}".format(thread_id), "{}_{}.jpg".format(i, string)))


@jit(nopython=False, nogil=True)
def make_multi_mjsynth(filenames, directory, num_data, image_size, sequence_length, num_retries, thread_id):

    os.mkdir(os.path.join(directory, "{}".format(thread_id)))

    for i in tqdm(range(num_data)):

        multi_image = np.zeros(image_size + [3], dtype=np.uint8)
        num_strings = random.randint(1, sequence_length)

        random_filenames = []
        random_rects = []

        for _ in range(num_strings):

            random_filename = random.sample(filenames, 1)
            image = cv2.imread(random_filename)

            for _ in range(num_retries):

                h = image.shape[0]
                w = image.shape[1]
                y = random.randint(0, image_size[0] - h)
                x = random.randint(0, image_size[1] - w)
                proposal = (y, x, y + h, x + w)

                for random_rect in random_rects:

                    if box(*proposal).intersects(box(*random_rect)):
                        break

                else:

                    multi_image[y:y+h, x:x+w, :] += image
                    random_filenames.append(random_filename)
                    random_rects.append(proposal)
                    break

        random_filenames = [random_filename for random_rect, random_filename in sorted(zip(random_rects, random_filenames))]
        strings = "_".join([os.path.splitext(os.path.basename(random_filename))[0].split("_")[1] for random_filename in random_filenames])

        cv2.imwrite(os.path.join(directory, "{}".format(thread_id), "{}_{}.jpg".format(i, strings)), multi_image)


if __name__ == "__main__":

    filenames = glob.glob("/home/sakuma/data/synth/*/*/*.jpg")
    random.shuffle(filenames)

    make_multi_thread(prepare_mjsynth, num_threads=32, split=True)(
        filenames[:int(len(filenames) * 0.9)],
        directory="/home/sakuma/data/mjsynth/train",
        image_size=[128, 128],
        string_length=10
    )

    make_multi_thread(prepare_mjsynth, num_threads=32, split=True)(
        filenames[int(len(filenames) * 0.9):],
        directory="/home/sakuma/data/mjsynth/test",
        image_size=[128, 128],
        string_length=10
    )

    filenames = glob.glob("/home/sakuma/data/mjsynth/train/*.jpg")

    make_multi_thread(make_multi_mjsynth, num_threads=32, split=False)(
        filenames,
        directory=os.path.dirname(filenames).replace("mjsynth, multi_mjsynth"),
        num_data=3000,
        image_size=[128, 128],
        sequence_length=4,
        num_retries=100
    )

    filenames = glob.glob("/home/sakuma/data/mjsynth/test/*.jpg")

    make_multi_thread(make_multi_mjsynth, num_threads=32, split=False)(
        filenames,
        directory=os.path.dirname(filenames).replace("mjsynth, multi_mjsynth"),
        num_data=300,
        image_size=[128, 128],
        sequence_length=4,
        num_retries=100
    )
