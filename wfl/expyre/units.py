"""Units-related utilities for converting between various time and memory formats
"""

import re

from functools import reduce


def time_to_HMS(t):
    return f'{int(t / 3600)}:{int(t / 60) % 60:02d}:{t % 60:02d}'


def time_to_sec(t):
    """convert time to seconds
    Parameters
    ----------
    t: str, int
        int: time in sec
        str format "float[sSmMhHdD]" number of seconds, minutes, hours, days, or "days(int)-HMS"
            or "HMS", with HMS being (HHHH:MM:SS | MM:SS | SS)
    Returns
    -------
    int time in seconds
    """
    if t is None or isinstance(t, int):
        return t

    t = t.strip().lower()

    if t == '_none_':
        return None

    m = re.search(r'^([0-9]+(?:\.[0-9]*)?)\s*([smhd])$', t)
    if m is not None:
        t = float(m.group(1))
        if m.group(2) == 'm':
            t *= 60
        elif m.group(2) == 'h':
            t *= 60 * 60
        elif m.group(2) == 'd':
            t *= 24 * 60 * 60

        return int(t)

    m = re.search(r'^(?:([0-9]+)-)?((?:[0-9]+:){0,2}(?:[0-9]+))$', t)
    if m is not None:
        if m.group(1) is not None:
            t_s = 24 * 60 * 60 * int(m.group(1))
        else:
            t_s = 0
        hms = m.group(2)
        # from https://stackoverflow.com/questions/6402812/how-to-convert-an-hmmss-time-string-to-seconds-in-python
        t_s += reduce(lambda sum, d: sum * 60 + int(d), hms.split(":"), 0)
        return t_s

    raise ValueError(f'Failed to parse time {t}')


def mem_to_kB(mem):
    """convert memory to kB
    Parameters
    ----------
    mem: str | int
        int: mem in kB
        str format "float[kKmMgGtT]b?": memory in KB, MB, GB, TB, float cannot have exponent

    Returns
    -------
    int memory in kB
    """
    if mem is None or isinstance(mem, int):
        return mem

    mem = mem.strip().lower()

    if mem == '_none_':
        return None

    m = re.search(r'^([0-9]+(?:\.[0-9]*)?)\s*([kmgt])b?$', mem)
    if m is not None:
        convs = {'k': 1, 'm': 1024, 'g': 1024**2, 't': 1024**3}
        return int(float(m.group(1)) * convs[m.group(2)])

    raise ValueError(f'Failed to parse memory {mem}')
