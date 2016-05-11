from lib.alexnet import AlexNet
from lib.data import Data
from lib.perf import Perf
import lib.utils as utils
import matplotlib.pyplot as plt
import lib.log as log
import numpy as np
import argparse
import os

# constants
IMAGES_DIR = 'data/Flickr_2800'

# actual run
if __name__ == "__main__":
    # parse args out
    parser = argparse.ArgumentParser()
    parser.add_argument("img_class", type=str, help="The directory from Flickr_2800 to plot")
    parser.add_argument("x", type=int, help="Columns")
    parser.add_argument("y", type=int, help="Rows")
    args = parser.parse_args()

    d = Data()
    d.load_images(os.path.join(IMAGES_DIR, args.img_class), None)
    d.graph(args.x, args.y)

    plt.show()
