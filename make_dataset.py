import numpy as np
import glob
import sys
import os
import cv2
import random
from tqdm import trange


def make_dataset(input_directory, output_directory, num_data, image_size, sequence_lengths, num_retries):

    input_filenames = glob.glob(os.path.join(input_directory, "*"))

    for i in trange(num_data):

        output_image = np.zeros(image_size + [3], dtype=np.uint8)

        strings = []
        rects = []

        for input_filename in random.sample(input_filenames, random.randint(1, sequence_lengths[0])):

            string = os.path.splitext(os.path.basename(input_filename))[0].split("_")[1]
            input_image = cv2.imread(input_filename)

            for _ in range(num_retries):

                h = input_image.shape[0]
                w = input_image.shape[1]
                y = random.randint(0, image_size[0] - h)
                x = random.randint(0, image_size[1] - w)
                proposal = (y, x, y + h, x + w)

                for rect in rects:
                    if box(*proposal).intersects(box(*rect)):
                        break

                else:
                    output_image[y:y+h, x:x+w, :] += input_image
                    strings.append(string)
                    rects.append(proposal)
                    break

        strings = [string for rect, string in sorted(zip(rects, strings))]
        output_filename = "{}_{}.jpg".format(i, "_".join(strings))
        cv2.imwrite(os.path.join(output_directory, output_filename), output_image)


if __name__ == "__main__":

    make_dataset(*sys.argv[1:3], int(sys.argv[3]), list(map(int, sys.argv[4:6])), list(map(int, sys.argv[6:8])), int(sys.argv[8]))
