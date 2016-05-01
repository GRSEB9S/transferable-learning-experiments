import datetime

def date_stamp():
    return str(datetime.datetime.now()).split('.')[0][8:].replace(' ', '_').replace(':', '-')
