import scipy.io
import cv2
from collections import Counter

dataset = scipy.io.loadmat("/home/sakuma/data/SynthText/gt.mat")
filenames = dataset["imnames"][0]
texts = dataset["txt"][0]
bounding_boxes = dataset["wordBB"][0]

num_data = len(filenames)
print("num_data: ".format(num_data))


def count(sequence_lengths):

    sequence_length_counter = Counter(sequence_lengths)
    total_num_sequences = len(sequence_lengths)
    partial_num_sequences = 0
    for sequence_length, num_sequences in sorted(sequence_length_counter.items()):
        partial_num_sequences += num_sequences
        ratio = partial_num_sequences / total_num_sequences
        if ratio > 0.95:
            print("max sequence length: {} (first over 90% of dataset ({}%))".format(sequence_length, int(ratio * 100)))
            break


sequence_lengths = [len(regions) for regions in texts]
count(sequence_lengths)

sequence_lengths = [len(strings.split("\n")) for regions in texts for strings in regions]
count(sequence_lengths)

sequence_lengths = [len(chars.strip(" ")) for regions in texts for strings in regions for chars in strings.split("\n")]
count(sequence_lengths)

chars = [char for regions in texts for strings in regions for chars in strings.split("\n") for char in chars]
char_counter = Counter(chars)

num_char_classes = len(char_counter)
print("num_char_classes: {}".format(num_char_classes))
print("chars: {}".format(sorted(char_counter.keys())))
