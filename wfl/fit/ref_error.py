import warnings
from pathlib import Path
from pprint import pprint
import json

import numpy as np

from wfl.configset import ConfigSet_out
from wfl.calculators.generic import run as generic_calc


def calc(input_configs, output_configs, calculator,
         ref_property_prefix, category_keys):
    """Calculate errors for some potential relative to some reference data.
    NOTE: several optional args such as properties and forces_by_components are not yet available.

    Parameters
    ----------
    input_configs: ConfigSet_in
        configs to check
    output_configs: ConfigSet_out
        optional location for calculated configs (otherwise will be in memory, and lost when function returns)
    calculator: tuple(constructor, args, kwargs)
        contructor for calculator with any args or kwargs
    ref_property_prefix: str
        prefix to property names for reference values
    category_keys: list(str)
        list of info keys to use to create categories that configs are grouped into

    Returns
    -------
    error: dict
        error for each quantity by group
    """

    # remove calc because multiprocessing.pool.map (used by the iterable_loop invoked in
    # calc) will send an Atoms object with pickle, and you can't pickle
    # an Atoms with a Potential attached as a calculator (weakref)
    input_configs = input_configs.in_memory()
    for at in input_configs:
        at.calc = None

    if output_configs is None:
        output_configs = ConfigSet_out()

    properties, calculator_properties = get_properties()

    # NOTE: should this, which requires storing all calculation results (in memory or a file), be the abstraction,
    # or should a single routine loop over atoms, do a calc, and accumulate error?
    # current approach nicely parallelizes using existing calculator parallelization over ConfigSet
    calculated_ats = generic_calc(input_configs, output_configs, calculator, properties=calculator_properties,
                                  output_prefix='calc_')

    return err_from_calculated_ats(calculated_ats, ref_property_prefix, calc_property_prefix='calc_',
                                   properties=properties, category_keys=category_keys)


def get_properties(properties=None):
    if properties is None:
        properties = ['energy_per_atom', 'forces', 'virial_per_atom']
    calculator_properties = []
    if 'energy' or 'energy_per_atom' in properties:
        calculator_properties.append('energy')
    if 'forces' in properties:
        calculator_properties.append('forces')
    if 'stress' or 'virial' or 'virial_per_atom' in properties:
        calculator_properties.append('stress')
    return properties, calculator_properties


def err_from_calculated_ats(calculated_ats, ref_property_prefix, calc_property_prefix,
                             properties=None, category_keys=None,
                             forces_by_component=False, forces_by_element=False):
    """calculate error for calculation results relative to stored reference values

    Parameters
    ----------
    calculated_ats: iterable(Atoms)
        input configurations with reference values and calculated in info/arrays
    ref_property_prefix: str
        prefix to info/array keys for reference properties
    calc_property_prefix: str
        prefix to info/array keys for calculated properties
    properties: list(str)
        list of 'energy', 'energy_per_atom', 'forces', 'stress', 'virial', 'virial_per_atom' to compute error for
    category_keys: str / list(str), default None
        results will be averaged by category, defined by a tuple containing the values of these
        keys in atoms.info, in addition to overall average _ALL_ category.
    forces_by_component: bool, default False
        define force error as difference between each component, rather than norm of vector difference
    forces_by_element: bool, default False
        calculate force error for each element separately

    Returns
    -------
        errors: dict of energy per atom, force, virial per atom rms errors for each category
    """

    properties, _ = get_properties(properties)

    if isinstance(category_keys, str):
        category_keys = [category_keys]
    if category_keys is not None and len(category_keys) == 0:
        category_keys = None

    all_errors = {None: {}}
    for at_i, at in enumerate(calculated_ats):
        if category_keys is None:
            category = None
        else:
            category = tuple([at.info.get(key, None) for key in category_keys])
        if category not in all_errors:
            all_errors[category] = {}

        at_errors = {}

        def _get(key, display_name, per_atom_prop):
            # simple getter for ref/calc properties
            try:
                if per_atom_prop:
                    return at.arrays[key]
                else:
                    return at.info[key]
            except KeyError:
                warnings.warn(
                    f'missing {display_name} property from config {at_i} '
                    f'type {at.info.get("config_type", None)}')
                return None

        if 'energy' in properties or 'energy_per_atom' in properties:
            e_ref = _get(ref_property_prefix + 'energy', "reference energy", False)
            e_calc = _get(calc_property_prefix + 'energy', "calculated energy", False)

            if e_ref is not None and e_calc is not None:
                e_error = e_calc - e_ref
                if 'energy' in properties:
                    at_errors['energy'] = [e_error]
                    at.info['energy_error'] = e_error
                if 'energy_per_atom' in properties:
                    at_errors['energy_per_atom'] = [e_error / len(at)]
                    at.info['energy_per_atom_error'] = e_error

        if 'forces' in properties:
            f_ref = _get(ref_property_prefix + 'forces', "reference forces", True)
            f_calc = _get(calc_property_prefix + 'forces', "calculated forces", True)

            if f_ref is not None and f_calc is not None:
                f_errors = f_calc - f_ref
                at.new_array('forces_error', f_errors)
                if forces_by_component:
                    f_errors = f_errors.reshape((-1))
                    atomic_numbers = np.asarray([[Z, Z, Z] for Z in at.numbers]).reshape((-1))
                else:
                    f_errors = np.linalg.norm(f_errors, axis=1)
                    atomic_numbers = at.numbers
                if forces_by_element:
                    at_errors['forces'] = {Z: f_errors[np.where(atomic_numbers == Z)[0]] for Z in set(at.numbers)}
                else:
                    at_errors['forces'] = f_errors

        if 'stress' in properties or 'virial' in properties or 'virial_per_atom' in properties:
            stress_ref = _get(ref_property_prefix + 'stress', "reference stress", False)
            stress_calc = _get(calc_property_prefix + 'stress', "calculated stress", False)

            if stress_ref is not None and stress_calc is not None:
                stress_errors = stress_calc - stress_ref
                vir_errors = -stress_errors * at.get_volume()
                if 'stress' in properties:
                    at_errors['stress'] = stress_errors
                    at.info['stress_error'] = stress_errors
                if 'virial' in properties:
                    at_errors['virial'] = vir_errors
                    at.info['virial_error'] = vir_errors
                if 'virial_per_atom' in properties:
                    at_errors['virial_per_atom'] = vir_errors / len(at)
                    at.info['virial_per_atom_error'] = vir_errors

        cats = [category]
        if category is not None:
            cats += [None]
        for cat in cats:
            for prop_k in at_errors:
                # at_errors[prop_k] can be list or dict of lists, add same struct to all_errors
                if prop_k not in all_errors[cat]:
                    if isinstance(at_errors[prop_k], dict):
                        all_errors[cat][prop_k] = {k: [] for k in at_errors[prop_k]}
                    else:
                        all_errors[cat][prop_k] = []
                # fill in list/dict in all_errors
                if isinstance(at_errors[prop_k], dict):
                    for k in at_errors[prop_k]:
                        if k not in all_errors[cat][prop_k]:
                            all_errors[cat][prop_k][k] = []
                        all_errors[cat][prop_k][k].extend(at_errors[prop_k][k])
                else:
                    all_errors[cat][prop_k].extend(at_errors[prop_k])

    # compute RMS
    for cat in all_errors.keys():
        for prop in all_errors[cat]:
            if isinstance(all_errors[cat][prop], dict):
                for k in all_errors[cat][prop]:
                    all_errors[cat][prop][k] = (len(all_errors[cat][prop][k]),
                                                np.sqrt(np.mean(np.asarray(all_errors[cat][prop][k]) ** 2)))
            else:
                all_errors[cat][prop] = (
                    len(all_errors[cat][prop]), np.sqrt(np.mean(np.asarray(all_errors[cat][prop]) ** 2)))

    # convert None key to '_ALL_'
    all_errors['_ALL_'] = all_errors[None]
    del all_errors[None]

    return all_errors
