import sys
from types import LambdaType

import numpy as np

from wfl.autoparallelize import autoparallelize, autoparallelize_docstring


def _select_autopara_wrappable(inputs, at_filter):
    """apply a filter to a sequence of configs

    Parameters
    ----------
    inputs: list(Atoms), ConfigSet
        input configurations
    at_filter: callable
        callable that takes an Atoms and returns a bool indicating if it should be selected
    """
    outputs = []
    for at in inputs:
        if at_filter(at):
            outputs.append(at)

    return outputs

def by_bool_func(*args, **kwargs):
    if "at_filter" in kwargs:
        at_filter = kwargs["at_filter"]
    else:
        at_filter = args[2]
    # disable parallelization by passing num_python_subprocesses=0
    if isinstance(at_filter, LambdaType) and at_filter.__name__ == "<lambda>":
        # turn of autoparallelization for lambdas, which cannot be pickled
        num_python_subprocesses = 0
    else:
        num_python_subprocesses = None
    default_autopara_info = {"num_python_subprocesses": num_python_subprocesses}
    return autoparallelize(_select_autopara_wrappable, *args,
           default_autopara_info=default_autopara_info, **kwargs)
autoparallelize_docstring(by_bool_func, _select_autopara_wrappable, "Atoms")

# NOTE this could probably be done with autoparallelize by returning a list with multiple
# copies when a single config needs to be returned multiple times, and either
# None or [] when a config isn't selected
def by_index(inputs, outputs, indices):
    """select atoms from configs by index

    Parameters
    ----------
    inputs: ConfigSet
        source configurations
    outputs: OutputSpec
        output configurations
    indices: list(int)
        Indices to be selected.  Values outside 0..len(inputs)-1 will be ignored.
        Repeated values will lead to multiple copies of configuration

    Returns
    -------
    ConfigSet pointing to selected configurations

    Notes
    -----
    This routine depends on details of ConfigSet and OutputSpec,
    so perhaps belongs as a use case of autoparallelize, but since it can return
    multiple outputs for a single input, this cannot be done right now
    """
    if outputs.all_written():
        sys.stderr.write('Returning before by_index since output is done\n')
        return outputs.to_ConfigSet()

    if len(indices) == 0:
        return outputs.to_ConfigSet()

    indices = sorted(indices)
    ii = iter(enumerate(inputs))
    try:
        at_i, at = next(ii)
    except StopIteration:
        return outputs.to_ConfigSet()
    cur_i = np.searchsorted(indices, at_i)
    while True:
        if cur_i >= len(indices):
            break
        if indices[cur_i] == at_i:
            outputs.store(at, at.info.pop("_ConfigSet_loc"))
            cur_i += 1
        else:
            try:
                at_i, at = next(ii)
            except StopIteration:
                break

    outputs.close()
    return outputs.to_ConfigSet()
