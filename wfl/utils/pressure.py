import numpy as np


class PressureRecursionError(BaseException):
    pass


def sample_pressure(pressure, at=None, rng=None):
    """Sample pressure for calculation with various modes

    Parameters
    ----------
    pressure: float / list / tuple
        Pressure, type and length defines mode as well
            - float: used as pressure
            - ("info", dict_key): looks for dict_key in at.info, parsed same as pressure argument here
            - ("exponential", float): exponential distribution, rate=1. and scaled by float given
            - ("normal_positive", mean, sigma): normal distribution with (mean, sigma) thrown away if negative value drawn,
              max 1000 tries
            - ("uniform", lower, upper): uniform distribution between bounds (lower, upper)

    at: ase.Atoms, default None
        atoms object, only needed or used if mode is `info`

    rng: numpy Generator object, default None
        random number generator to use. Only required if pressure will be generated randomly

    Returns
    -------
    p: float

    """
    try:
        if isinstance(pressure, (float, int)):
            p = pressure
        elif pressure[0] == 'info':
            if len(pressure) != 2:
                raise ValueError()
            # recursion on this to allow float and options below,
            # but corner case of at.info["key"] = ("info", "key") can lead to infinite recursion
            pressure_v = at.info[pressure[1]]
            if not isinstance(pressure_v, float) and pressure_v[0] == 'info' and at.info[pressure_v][1] == pressure[1]:
                raise PressureRecursionError('Infinite recursion in pressure {}'.format(pressure))

            p = sample_pressure(at.info[pressure[1]], at)
        elif pressure[0] == 'exponential':
            if len(pressure) != 2:
                raise ValueError()
            p = pressure[1] * rng.exponential(1.0)
        elif pressure[0] == 'normal_positive':
            if len(pressure) != 3:
                raise ValueError()
            n_try = 0
            p = -1.0
            while p < 0:
                n_try += 1
                if n_try >= 1000:
                    raise RuntimeError('Failed to get positive from normal distribution in 1000 iterations')
                p = rng.normal(pressure[1], pressure[2])
        elif pressure[0] == 'uniform':
            if len(pressure) != 3:
                raise ValueError()
            p = rng.uniform(pressure[1], pressure[2])
        else:
            raise ValueError()
    except ValueError as exc:
        raise RuntimeError('Failed to parse pressure \'{}\''.format(pressure)) from exc

    return p
