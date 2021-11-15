"""Miscellaneous utilities

This should be temporary and reorganised when there is more, or just make one utils file if there is not much.

"""

from ase import Atoms


def chunks(arr, n):
    """Yield successive n-sized chunks from arr

    Parameters
    ----------
    arr: list-like
    n: int
        length of chunks

    Yields
    ------
    arr_chunk: array_like
    """
    for i in range(0, len(arr), n):
        yield arr[i:i + n]


def atoms_to_list(atoms):
    if isinstance(atoms, Atoms):
        return [atoms]
    else:
        return atoms
