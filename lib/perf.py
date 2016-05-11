import csv
import matplotlib.pyplot as plt
import numpy as np

FIELD_NAMES = ['step', 'loss', 'accuracy']
FIT_DEGREE = 3

def float_values(d):
    return {k: float(v) for k, v in d.iteritems()}

class Perf(object):
    def __init__(self, data=None):
        self._data = data

    def get_data(self):
        return self._data

    def save(self, path):
        with open(path, 'w') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=FIELD_NAMES)
            writer.writeheader()
            writer.writerows(self._data)

    def restore(self, path):
        with open(path, 'r') as csvfile:
            reader = csv.DictReader(csvfile)
            self._data = [float_values(row) for row in reader]

    def _plot_key(self, key, color='r'):
        x = [perf['step'] for perf in self._data]
        y = [perf[key] for perf in self._data]

        self._graph_scatter(x, y, c=color)
        self._graph_best_fit(x, y, c=color)

    def plot_accuracy(self, color='r'):
        self._plot_key('accuracy', color=color)

    def plot_loss(self, color='r'):
        self._plot_key('loss', color=color)

    def _graph_scatter(self, x, y, c):
        plt.scatter(x, y, c=c)

    def _graph_best_fit(self, x, y, c):
        coeffs = np.polyfit(x, y, FIT_DEGREE)
        x_prime = np.arange(min(x)-1, max(x)+1, .01)
        y_prime = np.polyval(coeffs, x_prime)
        plt.plot(x_prime, y_prime, c=c)

    def graph(self):
        self.plot_accuracy('r')
        self.plot_loss('b')

        plt.show()
