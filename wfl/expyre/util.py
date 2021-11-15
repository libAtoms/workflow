import os


def remsh_cmd(cmd):
    if cmd is None:
        return os.environ.get('EXPYRE_RSH', 'ssh')
    else:
        return cmd
