import sys
from datasets import synth_text

if __name__ == "__main__":

    synth_text.convert_dataset(*sys.argv[1:-2], 4, 4, 4)