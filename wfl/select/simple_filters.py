import sys

import numpy as np
from ase import Atoms

from wfl.configset import ConfigSet
from wfl.autoparallelize import autoparallelize


class InfoAllIn:
    def __init__(self, fields_values):
        """filter operation (callable for simple_filters.apply()) that includes only
        configs where values of all specified Atoms.info fields are in the allowed list
        of values

        Parameters
        ----------
        fields_values: 2-tuple / list(2-tuples)
            list of tuples (info_field_name, allowed_values) such that
                Atoms.info[info_field_name] in allowed_values
        """
        if len(fields_values) == 2 and isinstance(fields_values[0], str):
            # promote single key-value pair to list
            fields_values = [fields_values]

        self.fields_values = fields_values

    def __call__(self, ats):
        results = []
        for at in ats:
            if all([at.info.get(info_field, None) in allowed_values for info_field, allowed_values in
                    self.fields_values]):
                results.append(at)
            else:
                results.append(None)
        return results


class InfoAllStartWith:
    def __init__(self, fields_values):
        """filter operation (callable for simple_filters.apply()) that includes only
        configs where values of all specified Atoms.info fields start with the
        specified strings

        Parameters
        ----------
        fields_values: 2-tuple / list(2-tuples)
            list of tuples (info_field_name, allowed_start) such that
                all of Atoms.info[info_field_name].startswith(allowed_start)
        """
        if len(fields_values) == 2 and isinstance(fields_values[0], str):
            # promote single key-value pair to list
            fields_values = [fields_values]

        self.fields_values = fields_values

    def __call__(self, ats):
        results = []
        for at in ats:
            if all([info_field in at.info and at.info[info_field].startswith(allowed_value) for
                    info_field, allowed_value in
                    self.fields_values]):
                results.append(at)
            else:
                results.append(None)
        return results


def apply(inputs, outputs, at_filter):
    """apply a filter to a sequence of configs

    Parameters
    ----------
    inputs: ConfigSet_in
        input configurations
    outputs: ConfigSet_out
        corresponding output configurations
    at_filter: callable
        callable that takes an iterable of Atoms and returns a list of selected Atoms

    Returns
    -------
    ConfigSet_in pointing to selected configurations

    """
    # disable parallelization by passing npool=0
    return autoparallelize(npool=0, iterable=inputs, configset_out=outputs, op=at_filter)


# NOTE this could probably be done with iterable_loop by returning a list with multiple
# copies when a single config needs to be returned multiple times, and either
# None or [] when a config isn't selected
def by_index(inputs, outputs, indices):
    """select atoms from configs by index

    Parameters
    ----------
    inputs: ConfigSet_in
        source configurations
    outputs: ConfigSet_out
        output configurations
    indices: list(int)
        Indices to be selected.  Values outside 0..len(inputs)-1 will be ignored.
        Repeated values will lead to multiple copies of configuration

    Returns
    -------
    ConfigSet_in pointing to selected configurations

    Notes
    -----
    This routine depends on details of ConfigSet_in and ConfigSet_out,
    so perhaps belongs as a use case of iterable_loop, but since it can return
    multiple outputs for a single input, this cannot be done eight now

    """
    if outputs.is_done():
        sys.stderr.write(f'Returning before by_index since output is done\n')
        return outputs.to_ConfigSet_in()

    if len(indices) == 0:
        return outputs.to_ConfigSet_in()

    indices = sorted(indices)
    ii = iter(enumerate(inputs))
    try:
        at_i, at = next(ii)
    except StopIteration:
        return outputs.to_ConfigSet_in()
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
    return outputs.to_ConfigSet_in()


def by_energy(inputs, outputs, lower_limit, upper_limit, energy_parameter_name=None, e0=None):
    """Filter by binding energy

    Parameters
    ----------
    inputs: ConfigSet_in
        source configurations
    outputs: ConfigSet_out
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
    ConfigSet_in pointing to selected configurations

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
        return outputs.to_ConfigSet_in()

    selected_any = False
    for atoms in inputs:
        if lower_limit < get_energy(atoms) < upper_limit:
            outputs.write(atoms)
            selected_any = True

    outputs.end_write()
    if selected_any:
        return outputs.to_ConfigSet_in()
    else:
        return ConfigSet(input_configs=[])
