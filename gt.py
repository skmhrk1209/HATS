import scipy.io
import cv2
from collections import Counter

dataset = scipy.io.loadmat("gt.mat")
filenames = dataset["imnames"][0]
texts = dataset["txt"][0]
bounding_boxes = dataset["wordBB"][0]

num_data = len(filenames)
print("num_data: ".format(num_data))

sequence_lengths = [len(sequence) for sequence in texts]
sequence_length_counter = Counter(sequence_lengths)

total_num_sequences = len(sequence_lengths)
partial_num_sequences = 0
for sequence_length, num_sequences in sorted(sequence_length_counter.items()):
    partial_num_sequences += num_sequences
    ratio = partial_num_sequences / total_num_sequences
    if ratio > 0.9:
        print("max sequence length: {} (first over 90% of dataset ({}%))".format(sequence_length, int(ratio * 100)))
        break

string_lengths = [len(string.strip()) for sequence in texts for string in sequence]
string_length_counter = Counter(string_lengths)

total_num_strings = len(string_lengths)
partial_num_strings = 0
for string_length, num_strings in sorted(string_length_counter.items()):
    partial_num_strings += num_strings
    ratio = partial_num_strings / total_num_strings
    if ratio > 0.9:
        print("max string length: {} (first over 90% of dataset ({}%))".format(string_length, int(ratio * 100)))
        break

chars = [char for sequence in texts for string in sequence for char in string]
char_counter = Counter(chars)

num_char_classes = len(char_counter)
print("num_char_classes: {}".format(num_char_classes))
print("chars: {}".format(sorted(char_counter.keys())))
