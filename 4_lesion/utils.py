import datetime
import numpy as np

def date_stamp():
    return str(datetime.datetime.now()).split('.')[0][8:].replace(' ', '_').replace(':', '-')

directories = ['bunny_2800', 'eye_2800', 'dolphin_2800', 'dog_2800']
