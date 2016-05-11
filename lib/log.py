import sys

def log(*args):
    print(' '.join([str(arg) for arg in args]))

def log_ephemeral(string):
    sys.stdout.write(string + '\r')
    sys.stdout.flush()
