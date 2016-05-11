import os
import datetime


def datestr():
    date, time = str(datetime.datetime.now()).split('.')[0][5:16].split(' ')
    return 'day' + date.replace('-', 'mon') + '_' + time.replace(':', 'h')


def ensure_path(path):
    if not os.path.exists(path):
        os.makedirs(path)


def euclid_dist(vector_list):
    """
    Get euclidian distance between most recent
    weight/bias vector and original weights/biases
    """
    return np.linalg.norm(vector_list[-1] - vector_list[0])
