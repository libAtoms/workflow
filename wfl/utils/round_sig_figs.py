"""
Rounding floats to significant figures
"""


def round_sig_figs(value, n_sig_figs):
    """Round to a certain number of significant figures

    based on:
    https://stackoverflow.com/questions/3410976/how-to-round-a-number-to-significant-figures-in-python

    Parameters
    ----------
    value: float
        value to round
    n_sig_figs: int
        number of significant figures

    Returns
    -------
    string representation of v, rounded to n_sig_figs significant figures
    """
    return '{:g}'.format(float('{:.{p}g}'.format(value, p=n_sig_figs)))
