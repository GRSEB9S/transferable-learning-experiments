import os
import datetime

def datestr():
    return str(datetime.datetime.now()).split('.')[0][8:].replace(' ', '_').replace(':', '-')

def ensure_path(path):
    if not os.path.exists(path):
        os.makedirs(path)
