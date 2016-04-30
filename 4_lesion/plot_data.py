import matplotlib.pyplot as plt
import numpy as np
import scipy.misc
import argparse
import glob
import os

NUM_X = 5
NUM_Y = 2

GRAY = True

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("path", help="Path to the data dir")
    args = parser.parse_args()

    images = glob.glob(os.path.join(args.path, "*.jpg"))

    if GRAY:
        plt.gray()

    for i in xrange(NUM_X * NUM_Y):
        plt.subplot(NUM_Y, NUM_X, i + 1)
        plt.imshow(scipy.misc.imread(images[i], flatten=True))

    plt.show()
