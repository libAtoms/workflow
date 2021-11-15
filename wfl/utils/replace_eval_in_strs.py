"""
Evaluation of expressions marked with _EVAL_ in strings, mainly used for interpreting config files.
"""

import warnings

from wfl.utils.round_sig_figs import round_sig_figs


def replace_eval_in_strs(obj, replacements, n_float_sig_figs=None):
    """Replace some string beginning with _EVAL_ in nested data structures
    with the result of eval() on them. Any lists, tuples, and dicts will
    be gone through recursively and replaced with substituted contents.
    Any strings starting with '_EVAL_ ' will be replaced with the result
    of an eval() call on the remainder of the string, after `replacements`
    has been used as the kwargs of a format() call.

    Parameters
    ----------
    obj: python object
        data structure to go through and replace '_EVAL_ ...' with return value of eval()
    replacements: dict
        keywords to format() call to be applied to each string before eval()
    n_float_sig_figs: int
        if not None, round float output of each eval to this many significant figures

    Returns
    -------
    obj: python object with new lists, tuples, and dicts, with _EVAL_ strings replaced by
        their eval() result
    """
    if isinstance(obj, str):
        if obj.startswith('_EVAL_ '):
            value = eval(obj.replace('_EVAL_ ', '', 1).format(**replacements))
            if n_float_sig_figs is not None and isinstance(value, float):
                value = float(round_sig_figs(value, n_float_sig_figs))
            return value
    elif isinstance(obj, list):
        return [replace_eval_in_strs(subobj, replacements, n_float_sig_figs) for subobj in obj]
    elif isinstance(obj, tuple):
        return (replace_eval_in_strs(subobj, replacements, n_float_sig_figs) for subobj in obj)
    elif isinstance(obj, dict):
        return {k: replace_eval_in_strs(v, replacements, n_float_sig_figs) for k, v in obj.items()}
    elif not isinstance(obj, (bool, int, float)):
        warnings.warn('replace_in_strings got unknown type {}, skipping replacement'.format(type(obj)))

    return obj
