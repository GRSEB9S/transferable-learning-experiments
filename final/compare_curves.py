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
END_ITER = 10000
STEP_SIZE = 250

def log_accuracy_linear_fit(extra_label, data):
    coeffs = np.polyfit([datum['step'] for datum in data], [datum['accuracy'] for datum in data], 1)
    log.log(extra_label, "Y-intercept: ", coeffs[0], " Slope: ", coeffs[1])

def get_step(data, step):
    for datum in data:
        if datum['step'] == step:
            return datum

def get_end_accuracy(data):
    end_step = get_step(data, END_ITER)
    return end_step['accuracy']

# returns -1 if not found
def get_line_reach(data, value):
    for i in range(STEP_SIZE, END_ITER + STEP_SIZE, STEP_SIZE):
        if get_step(data, i)['accuracy'] >= value:
            return i

    return -1

# actual run
if __name__ == "__main__":
    # parse args out
    parser = argparse.ArgumentParser()
    parser.add_argument("perfa", type=str, help="The identifier of the first perf curve to analyze")
    parser.add_argument("perfb", type=str, help="The identifier of the second perf curve to analyze")
    args = parser.parse_args()

    # restore curves
    pa = Perf()
    patha = os.path.join(PERF_DIR, args.perfa + '.csv')
    pa.restore(patha)

    pb = Perf()
    pathb = os.path.join(PERF_DIR, args.perfb + '.csv')
    pb.restore(pathb)

    # load data in
    pa_data = pa.get_data()
    pb_data = pb.get_data()

    # print ends
    log.log("FINAL ACCURACY")
    log.log("===")
    log.log("Perf A End Accuracy: ", get_end_accuracy(pa_data))
    log.log("Perf B End Accuracy: ", get_end_accuracy(pb_data))
    log.log("")

    # figure out linear coeffs
    log.log("LINEAR COEFFS CURVES")
    log.log("===")
    log_accuracy_linear_fit("Perf A Accuracy Best-fit Line: ", pa_data)
    log_accuracy_linear_fit("Perf B Accuracy Best-fit Line: ", pb_data)
    log.log("")

    # create joint data to compute integral
    joint_data = {}
    for data, key in [(pa_data, 'a'), (pb_data, 'b')]:
        for datum in data:
            if datum['step'] not in joint_data:
                joint_data[datum['step']] = {
                    'a': None,
                    'b': None
                }

            joint_data[datum['step']][key] = datum

    # sanity check
    assert(len(joint_data) == END_ITER/STEP_SIZE)

    # figure out how long it took to get there
    log.log("REACH")
    log.log("===")
    log.log("It took curve B ", get_line_reach(pb_data, get_end_accuracy(pa_data)), " iterations to reach curve A's end accuracy.")
    log.log("")

    # compute integral (right hand riemann sum)
    rhs = 0
    for i in range(STEP_SIZE, END_ITER + STEP_SIZE, STEP_SIZE):
        rhs += (get_step(pb_data, i)['accuracy'] - get_step(pa_data, i)['accuracy']) * STEP_SIZE

    log.log("RHS of Accuracy")
    log.log("===")
    log.log(rhs)
    log.log("")
