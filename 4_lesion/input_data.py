from __future__ import print_function
import gzip
import os
import urllib
import numpy as np
import scipy.misc
import glob
import math


def extract_images(positive_dir, negative_dir):
    
    # iterate through all files
    data = []
    labels = []

    for directory, classification in [(positive_dir, np.array([1, 0])), (negative_dir, np.array([0, 1]))]:
        images = glob.glob(os.path.join(directory, "*.jpg"))

        for image in images:
            data.append(scipy.misc.imread(image, flatten=True))
            labels.append(classification)

    data = np.array(data)
    data = data.reshape(data.shape[0], data.shape[1], data.shape[2], 1)
    return data, np.array(labels)


class DataSet(object):
  def __init__(self, images, labels, fake_data=False):
    assert images.shape[0] == labels.shape[0], ("images.shape: %s labels.shape: %s" % (images.shape, labels.shape))
    self._num_examples = images.shape[0]
    # Convert shape from [num examples, rows, columns, depth]
    # to [num examples, rows*columns] (assuming depth == 1)
    assert images.shape[3] == 1
    images = images.reshape(images.shape[0],images.shape[1] * images.shape[2])
    # Convert from [0, 255] -> [0.0, 1.0].
    images = images.astype(np.float32)
    images = np.multiply(images, 1.0 / 255.0)

    self._images = images
    self._labels = labels
    self._epochs_completed = 0
    self._index_in_epoch = 0

    self._shuffle_data()


  @property
  def images(self):
    return self._images


  @property
  def labels(self):
    return self._labels


  @property
  def num_examples(self):
    return self._num_examples


  @property
  def epochs_completed(self):
    return self._epochs_completed


  def _shuffle_data(self):
      perm = np.arange(self._num_examples)
      np.random.shuffle(perm)
      self._images = self._images[perm]
      self._labels = self._labels[perm]


  def next_batch(self, batch_size, fake_data=False):
    """Return the next `batch_size` examples from this data set."""
    start = self._index_in_epoch
    self._index_in_epoch += batch_size

    if self._index_in_epoch > self._num_examples:
        # Finished epoch
        self._epochs_completed += 1

        # shuffle data
        self._shuffle_data()

        # Start next epoch
        start = 0
        self._index_in_epoch = batch_size
        assert batch_size <= self._num_examples

    end = self._index_in_epoch
    return self._images[start:end], self._labels[start:end]


def read_data_sets(positive_dir, negative_dir):
    class DataSets(object):
        pass
    data_sets = DataSets()

    # get data
    images, labels = extract_images(positive_dir, negative_dir)

    # shuffle data
    total_images = len(images)

    perm = np.arange(total_images)
    np.random.shuffle(perm)
    images = images[perm]
    labels = labels[perm]

    # create the split
    train_split = 0.6
    training_images = int(math.ceil(total_images * train_split))

    # create sets
    data_sets.train = DataSet(images[:training_images], labels[:training_images])
    data_sets.test = DataSet(images[training_images:], labels[training_images:])

    return data_sets

def show_a_few(dataset):
    for i in xrange(5):
        print(str(dataset._labels[i]))
        import matplotlib.pyplot as plt
        plt.imshow(dataset._images[i].reshape(40, 40))
        plt.show()

if __name__ == "__main__":
    a = read_data_sets('./data/car', './data/notcar')
    show_a_few(a.train)
    show_a_few(a.test)
