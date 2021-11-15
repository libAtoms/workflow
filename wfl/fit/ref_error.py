import warnings

import numpy as np

from wfl.calculators.generic import run as calc_run


def calc(input_configs, output_configs, calculator, ref_property_prefix, calc_property_prefix='calc_',
         properties=None, category_keys=None,
         forces_by_component=False, forces_by_element=False):
    """calculate error for calculation results relative to stored reference values

    Parameters
    ----------
    input_configs: ConfigSet_in
        input configurations with reference values in info/arrays
    output_configs: ConfigSet_out
        storage for configs with calculator results
    calculator: tuple / Calculator
        calculator to evaluate, as required by wfl.calculators.generic.run()
    ref_property_prefix: str
        prefix to info/array keys for reference properties
    calc_property_prefix: str, default 'calc_'
        prefix to info/array keys for calculated properties
    properties: list(str)
        list of 'energy', 'energy_per_atom', 'forces', 'stress', 'virial', 'virial_per_atom' to compute error for
    category_keys: str / list(str)
        results will be averaged by category, defined by a tuple containing the values of these
        keys in atoms.info, in addition to overall average None category.
    forces_by_component: bool, default False
        define force error as difference between each component, rather than norm of vector difference
    forces_by_element: bool, default False
        calculate force error for each element separately

    Returns
    -------
        errors: dict of energy per atom, force, virial per atom rms errors for each category
    """
    properties, calculator_properties = process_properties(properties)

    if calc_property_prefix is None:
        raise ValueError('Reference error calculator cannot use SinglePointCalculator '
                         'calculated properties (calc_property_prefix is None)')

    # NOTE: should this, which requires storing all calculation results (in memory or a file), be the abstraction,
    # or should a single routine loop over atoms, do a calc, and accumulate error?
    # current approach nicely parallelizes using existing calculator parallelization over ConfigSet
    calculated_ats = calc_run(input_configs, output_configs, calculator, properties=calculator_properties,
                              output_prefix=calc_property_prefix)

    return calc_from_calculated_ats(calculated_ats, ref_property_prefix, calc_property_prefix=calc_property_prefix,
                                    properties=None, category_keys=category_keys,
                                    forces_by_component=forces_by_component, forces_by_element=forces_by_element)


def process_properties(properties):
    if properties is None:
        properties = ['energy_per_atom', 'forces', 'virial_per_atom']
    assert all(
        [p in ['energy', 'energy_per_atom', 'forces', 'stress', 'virial', 'virial_per_atom'] for p in properties])
    calculator_properties = []
    if 'energy' or 'energy_per_atom' in properties:
        calculator_properties.append('energy')
    if 'forces' in properties:
        calculator_properties.append('forces')
    if 'stress' or 'virial' or 'virial_per_atom' in properties:
        calculator_properties.append('stress')
    return properties, calculator_properties


def calc_from_calculated_ats(calculated_ats, ref_property_prefix, calc_property_prefix,
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

    properties, _ = process_properties(properties)

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

        def _get(key, display_name, per_atom):
            # simple getter for ref/calc properties
            try:
                if per_atom:
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
                if 'energy_per_atom' in properties:
                    at_errors['energy_per_atom'] = [e_error / len(at)]

        if 'forces' in properties:
            f_ref = _get(ref_property_prefix + 'forces', "reference forces", True)
            f_calc = _get(calc_property_prefix + 'forces', "calculated forces", True)

            if f_ref is not None and f_calc is not None:
                f_errors = f_calc - f_ref
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
                v_errors = -stress_errors * at.get_volume()
                if 'stress' in properties:
                    at_errors['stress'] = stress_errors
                if 'virial' in properties:
                    at_errors['virial'] = v_errors
                if 'virial_per_atom' in properties:
                    at_errors['virial_per_atom'] = v_errors / len(at)

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


def clean_up_for_json(error_dict):
    """Cleans error dictionary to be JSON serializable

    Builds on format of calc above, converts the tuple keys to strings

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
            k = '(' + ','.join([str(k_el) for k_el in k]) + ')'
        error_dict_json_compatible[k] = v
    return error_dict_json_compatible
