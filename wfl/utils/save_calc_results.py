import re
import warnings

from ase.outputs import ArrayProperty, all_outputs
from ase.calculators.singlepoint import SinglePointCalculator
from ase.calculators.calculator import all_properties, PropertyNotImplementedError


# Following ase.io.extxyz
# Determine 'per-atom' and 'per-config' based on all_outputs shape,
# but filter for things in all_properties because that's what
# SinglePointCalculator accepts
per_atom_properties = []
per_config_properties = []
for key, val in all_outputs.items():
    if key not in all_properties:
        continue
    if isinstance(val, ArrayProperty) and val.shapespec[0] == 'natoms':
        per_atom_properties.append(key)
    else:
        per_config_properties.append(key)


def save_calc_results(atoms, *, prefix, properties):
    """saves results of a calculation in a SinglePointCalculator or info/arrays keys

    If atoms.info["__calculator_results_saved"] is true, assume that results have already been saved
    and instead just remove this key and continue

    Parameters
    ----------
    atoms: Atoms
        configuration
    properties: list(str) or None
        list of calculated properties to save, or None to use atoms.calc.results dict keys
    prefix: str / None , default None
        if None, store in SinglePointCalculator, else store in prefix+<property>.
        str with length 0 is forbidden
    """
    if atoms.info.pop("__calculator_results_saved", False):
        return

    if isinstance(prefix, str) and len(prefix) == 0:
        raise ValueError('Refusing to save calculator results into info/arrays fields with no prefix,'
                         ' too much chance of confusion with ASE extxyz reading/writing and conversion'
                         ' to SinglePointCalculator')

    if properties is None:
        # This will not work for calculators like Castep that (as of some point
        # at least) do not use results dict.  Such calculators will fail below,
        # in the "if 'energy' in properties" statement.

        properties = list(atoms.calc.results.keys())

    if prefix is not None:
        for p in per_config_properties:
            if prefix + p in atoms.info:
                del atoms.info[prefix + p]
        for p in per_atom_properties:
            if prefix + p in atoms.arrays:
                del atoms.arrays[prefix + p]

    # copy per-config and per-atom results
    config_results = {}
    atoms_results = {}
    for prop_name in properties:
        if prop_name not in atoms.calc.results:
            from ase.calculators.calculator import PropertyNotPresent
            raise PropertyNotPresent(f"{prop_name} is not one of the calculated properties "
                                     f"({atoms.calc.results.keys()}).")
        if prop_name == 'energy':
            try:
                config_results['energy'] = atoms.get_potential_energy(force_consistent=True)
            except PropertyNotImplementedError:
                config_results['energy'] = atoms.get_potential_energy()
            continue
        if prop_name in per_config_properties:
            config_results[prop_name] = atoms.calc.results.get(prop_name)
        if prop_name in per_atom_properties:
            atoms_results[prop_name] = atoms.calc.results.get(prop_name)
    try:
        if prefix is not None:
            config_results['converged'] = atoms.calc.converged
    except AttributeError as exc:
        pass
    if "extra_results" in dir(atoms.calc):
        if prefix is None and (len(atoms.calc.extra_results.get("config", {})) > 0 or
                                       len(atoms.calc.extra_results.get("atoms", {})) > 0):
            raise ValueError('Refusing to save calculator results into info/arrays fields with no prefix,'
                            ' too much chance of confusion with ASE extxyz reading/writing and conversion'
                            ' to SinglePointCalculator')

        for key, vals in atoms.calc.extra_results["config"].items():
            config_results[key] = vals

        for key, vals in atoms.calc.extra_results["atoms"].items():
            atoms_results[key] = vals

        # Update atoms' positions if geometry was optimised
        if "relaxed_positions" in atoms.calc.extra_results:
            atoms.set_positions(atoms.calc.extra_results["relaxed_positions"])

    # write to Atoms
    if prefix is None:
        # Filter out nonstandard properties that will cause SinglePointCalculator to fail
        config_results = {k: v for k, v in config_results.items() if k in all_properties}
        atoms_results = {k: v for k, v in atoms_results.items() if k in all_properties}
        atoms.calc = SinglePointCalculator(atoms, **config_results, **atoms_results)
    else:
        atoms.calc = None
        for p, v in config_results.items():
            atoms.info[prefix + p] = v
        for p, v in atoms_results.items():
            atoms.new_array(prefix + p, v)


def at_copy_save_calc_results(at, prefix, properties=None):
    """Make a copy of an atoms object with calculator results saved in
    info/arrays keys.  If prefix starts with "<common_prefix>__", all
    other properties starting with "<common_prefix>__" will be deleted.

    Parameters
    ----------
    at: Atoms
        source Atoms object
    prefix: str / None
        prefix to prepend to properties, None for storing in SinglePointCalculator
    properties: list, default None
        list of properties to store

    Returns
    -------
    at_copy: copied Atoms with calc results stored
    """

    at_copy = at.copy()
    at_copy.calc = at.calc

    if prefix is not None and "__" in prefix:
        common_prefix = re.sub(r"__.*", "__", prefix)
        for k in list(at_copy.info):
            if k.startswith(common_prefix):
                del at_copy.info[k]
        for k in list(at_copy.arrays):
            if k.startswith(common_prefix):
                del at_copy.arrays[k]

    save_calc_results(at_copy, prefix=prefix, properties=properties)

    return at_copy
