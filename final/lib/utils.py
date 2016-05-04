import os
import datetime


def datestr():
    date, time = str(datetime.datetime.now()).split('.')[0][5:16].split(' ')
    return 'day' + date.replace('-', 'mon') + '_' + time.replace(':', 'h')


def ensure_path(path):
    if not os.path.exists(path):
        os.makedirs(path)
