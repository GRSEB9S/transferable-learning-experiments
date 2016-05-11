import argparse
import numpy as np
import matplotlib.pyplot as plt
import csv

GRAPH = "train_loss"
DEG = 3 # degree of best fit curve

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("path", help="Path to the perf file")
    args = parser.parse_args()

    with open(args.path) as csvfile:
        perfs = csv.DictReader(csvfile)

        # get scatter plot data
        x = []
        y = []

        for perf in perfs:
            x.append(float(perf['iteration']))
            y.append(float(perf[GRAPH]))

        plt.scatter(x, y)

        # create best fit curve
        coeffs =np.polyfit(x, y, DEG)
        x2 = np.arange(min(x)-1, max(x)+1, .01) #use more points for a smoother plot
        y2 = np.polyval(coeffs, x2) #Evaluates the polynomial for each x2 value
        plt.plot(x2, y2)

        # show
        plt.show()
