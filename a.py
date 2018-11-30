import glob
import cv2
from tqdm import tqdm

fs = glob.glob("/home/sakuma/data/synth/*/*")

for f in tqdm(fs):
    cv2.imread(f)