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

        threads = [threading.Thread(
            target=func,
            args=args,
            kwargs=dict(kwargs, thread_id=i)
        ) for i in range(num_threads)]

        for thread in threads:
            thread.start()

        for thread in threads:
            thread.join()

    return func_mt


@jit(nopython=False, nogil=True)
def make_multi_mjsynth(filenames, directory, num_data, image_size, sequence_length, string_length, num_retries, thread_id):

    for i in tqdm(range(num_data * thread_id, num_data * (thread_id + 1))):

        multi_image = np.zeros(image_size + [3], dtype=np.uint8)
        num_strings = random.randint(1, sequence_length)

        random_filenames = []
        random_rects = []

        for _ in range(num_strings):

            while True:

                random_filename = random.sample(filenames, 1)[0]
                string = os.path.splitext(os.path.basename(random_filename))[0].split("_")[1]

                if len(string) <= string_length:

                    image = cv2.imread(random_filename)

                    if image is not None and image.shape[:2] <= image_size:
                        break

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

        cv2.imwrite(os.path.join(directory, "{}_{}.jpg".format(i, strings)), multi_image)


if __name__ == "__main__":

    filenames = glob.glob("/home/sakuma/data/mjsynth/*/*/*.jpg")

    random.seed(0)
    random.shuffle(filenames)

    make_multi_thread(make_multi_mjsynth, num_threads=32)(
        filenames=filenames[:int(len(filenames) * 0.9)],
        directory="/home/sakuma/data/multi_mjsynth/train",
        num_data=3000,
        image_size=[128, 128],
        sequence_length=4,
        string_length=10,
        num_retries=100
    )

    make_multi_thread(make_multi_mjsynth, num_threads=32)(
        filenames=filenames[int(len(filenames) * 0.9):],
        directory="/home/sakuma/data/multi_mjsynth/test",
        num_data=300,
        image_size=[128, 128],
        sequence_length=4,
        string_length=10,
        num_retries=100
    )
