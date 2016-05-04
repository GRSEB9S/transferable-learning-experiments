from __future__ import print_function
import gzip
import os
import urllib
import numpy as np
import scipy.misc
import glob
import math
import random
import sys

IMG_WIDTH = 40
IMG_HEIGHT = 40


class Data(object):
    def __init__(self, data=None):
        # data source
        if data is None:
            data = []
        self._data = data

        # internal book keeping for batchings
        self._index_in_epoch = 0

    def load_images(self, path, label):
        # iterate through all files
        images = glob.glob(os.path.join(path, "*.jpg"))

        # sanity check
        if len(images) == 0:
            print("Missing path", path)
            sys.exit()

        # load images in
        for image in images:
            # read in image
            image_data = scipy.misc.imread(image, flatten=True)

            # convert [0, 255] pixel values to [0, 1]
            image_data.astype(np.float32)
            image_data = np.multiply(image_data, 1.0 / 255.0)

            # reshape from 2d matrix to 1d vector
            image_data = image_data.reshape(IMG_WIDTH * IMG_HEIGHT)

            # add it to the data set
            self._data.append((image_data, label))

        # shuffle the data
        self.shuffle()

    def shuffle(self):
        random.shuffle(self._data)

    def split_train_test(self, train_split=0.5):
        cutoff = int(math.ceil(len(self._data) * train_split))
        return (
            Data(data=self._data[:cutoff]),
            Data(data=self._data[cutoff:])
        )

    def get_all(self):
        return self.x(0, len(self._data)), self.y(0, len(self._data))

    def x(self, start, end):
        return np.array([datum[0] for datum in self._data[start:end]])

    def y(self, start, end):
        return np.array([datum[1] for datum in self._data[start:end]])

    def get_batch(self, size):
        # sanity check
        assert size <= len(self._data)

        # move the pointer forward
        start = self._index_in_epoch
        self._index_in_epoch += size

        if self._index_in_epoch > len(self._data):
            # shuffle data
            self.shuffle()

            # Start next epoch
            start = 0
            self._index_in_epoch = size

        end = self._index_in_epoch
        return self.x(start, end), self.y(start, end)

    def graph(self, x, y):
        for i in xrange(x * y):
            plt.subplot(y, x, i + 1)
            plt.imshow(self._data[i][0].reshape(IMG_WIDTH, IMG_HEIGHT))
