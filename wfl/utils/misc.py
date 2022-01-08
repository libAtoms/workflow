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


def dict_tuple_keys_to_str(error_dict):
    """Convert tuple keys to strings so Dict is JSON serializable

    Parameters
    ----------
    error_dict: dict

    Returns
    -------
    error_dict_json_compatible: dict
    """
    error_dict_json_compatible = {}
    for k, v in error_dict.items():
        if isinstance(k, tuple):
            k = '(' + ','.join([str(kv) for kv in k]) + ')'
        error_dict_json_compatible[k] = v
    return error_dict_json_compatible
