import tensorflow as tf
import numpy as np
import glob
import sys
import os


def multi_thread(func, num_threads):

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
def make_dataset(input_filenames, output_directory, num_data, image_size, sequence_lengths, num_retries, thread_id):

    for i in trange(num_data * thread_id, num_data * (thread_id + 1)):

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

    multi_thread(make_dataset, num_threads=os.cpu_count())(
        input_filenames=glob.glob(os.path.join(sys.argv[1], "*")),
        output_directory=sys.argv[2],
        num_data=int(sys.argv[3]),
        image_size=[256, 256],
        sequence_lengths=[4, 10],
        num_retries=100
    )
