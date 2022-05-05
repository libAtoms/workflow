import sys
from types import LambdaType

import numpy as np
from ase import Atoms

from wfl.configset import ConfigSet
from wfl.autoparallelize import autoparallelize


def select(inputs, outputs, at_filter):
    """apply a filter to a sequence of configs

    Parameters
    ----------
    inputs: ConfigSet
        input configurations
    outputs: OutputSpec
        corresponding output configurations
    at_filter: callable
        callable that takes an Atoms and returns a bool indicating if it should be selected

    Returns
    -------
    ConfigSet pointing to selected configurations
    """
    # disable parallelization by passing npool=0
    if isinstance(at_filter, LambdaType) and at_filter.__name__ == "<lambda>":
        # turn of autoparallelization for lambdas, which cannot be pickled
        npool = 0
    else:
        npool = None
    return autoparallelize(npool=npool, iterable=inputs, outputspec=outputs, at_filter=at_filter, op=_select_autopara_wrappable)


def _select_autopara_wrappable(inputs, at_filter):
    outputs = []
    for at in inputs:
        if at_filter(at):
            outputs.append(at)

    return outputs


# NOTE this could probably be done with iterable_loop by returning a list with multiple
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
    so perhaps belongs as a use case of iterable_loop, but since it can return
    multiple outputs for a single input, this cannot be done right now
    """
    if outputs.is_done():
        sys.stderr.write(f'Returning before by_index since output is done\n')
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
            outputs.write(at, from_input_file=inputs.get_current_input_file())
            cur_i += 1
        else:
            try:
                at_i, at = next(ii)
            except StopIteration:
                break

    outputs.end_write()
    return outputs.to_ConfigSet()


# DEPRECATE - messy, only used by wfl/cli/reactions_iter_fit.py, and can probably be done using select() above
def by_energy(inputs, outputs, lower_limit, upper_limit, energy_parameter_name=None, e0=None):
    """Filter by binding energy

    Parameters
    ----------
    inputs: ConfigSet
        source configurations
    outputs: OutputSpec
        output configurations
    lower_limit: float / None
        lower energy limit for binding energy, None is -inf
    upper_limit: float / None
        upper energy limit for binging energy, None is +inf
    energy_parameter_name: str / None, default None
        parameter name to use for energy, if None then atoms.get_potential_energy() is used
    e0 : dict / None
        energy of isolated atoms, to use for binding energy calculation, with chemical symbols as keys
        None triggers all zero

    Returns
    -------
    ConfigSet pointing to selected configurations

    """

    if lower_limit is None:
        lower_limit = - np.inf

    if upper_limit is None:
        upper_limit = np.inf

    def get_energy(at: Atoms):
        if e0 is None:
            shift = 0.
        else:
            shift = np.sum([e0[symbol] for symbol in at.get_chemical_symbols()])

        if energy_parameter_name is None:
            return (at.get_potential_energy() - shift) / len(at)
        else:
            return (at.info.get(energy_parameter_name) - shift) / len(at)

    if outputs.is_done():
        sys.stderr.write(f'Returning before by_energy since output is done\n')
        return outputs.to_ConfigSet()

    selected_any = False
    for atoms in inputs:
        if lower_limit < get_energy(atoms) < upper_limit:
            outputs.write(atoms)
            selected_any = True

    outputs.end_write()
    if selected_any:
        return outputs.to_ConfigSet()
    else:
        return ConfigSet(input_configs=[])
