import csv
import matplotlib.pyplot as plt
import numpy as np

FIELD_NAMES = ['step', 'loss', 'accuracy']
FIT_DEGREE = 3

class Perf(object):
    def __init__(self, data=None):
        self._data = data

    def save(self, path):
        with open(path, 'w') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=FIELD_NAMES)
            writer.writeheader()
            writer.writerows(self._data)

    def restore(self, path):
        with open(path, 'w') as csvfile:
            reader = csv.DictReader(csvfile, fieldnames=FIELD_NAMES)
            self._data = [row for row in reader]

    def _graph_scatter(self, x, y, c):
        plt.scatter(x, y, c=c)

    def _graph_best_fit(self, x, y, c):
        coeffs = np.polyfit(x, y, FIT_DEGREE)
        x_prime = np.arange(min(x)-1, max(x)+1, .01)
        y_prime = np.polyval(coeffs, x_prime)
        plt.plot(x_prime, y_prime)

    def graph(self):
        # get scatter plot data
        x = []
        y1 = []  # accuracy
        y2 = []  # loss

        for perf in self._data:
            x.append(float(perf['step']))
            y1.append(float(perf['accuracy']))
            y2.append(float(perf['loss']))

        self._graph_scatter(x, y1, '#990000')
        self._graph_scatter(x, y2, '#000099')

        self._graph_best_fit(x, y1, '#FF0000')
        self._graph_best_fit(x, y2, '#000099')

        plt.show()
