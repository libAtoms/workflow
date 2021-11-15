import sys
from datetime import datetime


def print_log(msg, show_time=True, logfile=sys.stdout):
    if show_time:
        msg += ' ' + datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    logfile.write(msg + '\n')
    logfile.flush()


def increment_active_iter(active_iter):
    try:
        with open('ACTIVE_ITER') as fin:
            file_active_iter = int(fin.readline())
    except FileNotFoundError:
        file_active_iter = None
    if file_active_iter is not None and active_iter + 1 > file_active_iter:
        # file exists and incrementing past its value
        with open('ACTIVE_ITER', 'w') as fout:
            fout.write('{}\n'.format(active_iter + 1))


def process_active_iter(active_iter):
    if active_iter is None:
        # read from file
        try:
            with open('ACTIVE_ITER') as fin:
                active_iter = int(fin.readline())
        except FileNotFoundError:
            # initialize file
            active_iter = 0
            with open('ACTIVE_ITER', 'w') as fout:
                fout.write('{}\n'.format(active_iter))

    return active_iter
