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
PERF_DIR = 'results/perf'

# main logic
def graph_perf(name, accuracy, color='r'):
    p = Perf()
    path = os.path.join(PERF_DIR, name + '.csv')
    p.restore(path)

    if accuracy:
        p.plot_accuracy(color=color)
    else:
        p.plot_loss(color=color)

# actual run
if __name__ == "__main__":
    # parse args out
    parser = argparse.ArgumentParser()
    parser.add_argument("perf", type=str, help="The identifier of the perf curve to graph")
    parser.add_argument("-s", "--secondperf", type=str, help="The identifier of a second perf curve to graph")
    parser.add_argument('-a', '--accuracy', action='store_true', help='Whether to plot accuracy instead of loss', default=False)
    args = parser.parse_args()

    # graph perf
    graph_perf(args.perf, args.accuracy, color='r')

    # graph second one if exists
    if args.secondperf is not None:
        graph_perf(args.secondperf, args.accuracy, color='b')

    plt.show()
