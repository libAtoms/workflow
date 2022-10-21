import warnings
import re

import numpy as np

from wfl.configset import ConfigSet, OutputSpec
from wfl.calculators.generic import run as generic_calc


def calc(inputs, calc_property_prefix, ref_property_prefix,
         config_properties=None, atom_properties=None, category_keys="config_type",
         weight_property=None, calc_outputs=None, calc_autopara_info=None):
    """calculate error for calculation results relative to stored reference values

    Parameters
    ----------
    inputs: iterable(Atoms)
        input configurations with reference and calculated values in Atoms.info/arrays
    calc_property_prefix: str / Calculator / tuple(calculator_constructor, args, kwargs)
        prefix to info/array keys for calculated properties, or calculator constructor
    ref_property_prefix: str
        prefix to info/array keys for reference properties
    config_properties: list(str), default ["energy/atom", "virial/atom"]
        list of ``Atoms.info`` calculated properties (to be prefixed by ``ref_property_prefix``
        or ``calc_property_prefix``) to compute error for.  ``virial`` will be reconstructed from ``stress``.
        Properties can end with ``/atom`` or ``/comp`` for different components being counted separately.
        Default only used if neither ``config_properties`` nor ``atom_properties`` is present.
    atom_properties: list(str), default ["forces"]
        list of `Atoms.arrays` calculated properties (to be prefixed by ``ref_property_prefix``
        or ``calc_property_prefix``) to compute error For.  Properties can end with ``/comp``
        for different components being computed separately, and ``/Z`` for different atomic numbers
        to be assigned to different categories.  Default only used if neither ``config_properties``
        nor ``atom_properties`` is present.
    category_keys: str / list(str), default "config_type"
        results will be averaged by category, defined by a string containing the values of these
        keys in Atoms.info, in addition to overall average _ALL_ category.
    weight_property: str, optional
        if present, Atoms.info key for weights to apply to RMSE calculation
    calc_outputs: OutputSpec, optional
        where to store configs with calculated properties from optional calculation of
        properties to be tested
    calc_autopara_info: AutoparaInfo, optional
        autoparallelization info for optional initial calculation of properties to be tested

    Returns
    -------
        errors: dict of energy per atom, force, virial per atom rms errors for each category
    """

    # default properties
    if config_properties is None and atom_properties is None:
        config_properties = ["energy/atom", "virial/atom"]
        atom_properties = ["forces"]
    if config_properties is None:
        config_properties = []
    if atom_properties is None:
        atom_properties = []

    # optionally calculate properties to be compared to reference
    if not isinstance(calc_property_prefix, str):
        calculator_properties = [re.sub(r"^virial$", "stress", re.sub(r"/atom\b", "", re.sub(r"/comp\b", "", re.sub(r"/Z\b", "", prop)))) for
                                 prop in config_properties + atom_properties]
        generic_calc_kwargs = {}
        if calc_autopara_info is not None:
            generic_calc_kwargs = { "autopara_info": calc_autopara_info }
        inputs = generic_calc(inputs, calc_outputs, calc_property_prefix, properties=calculator_properties, output_prefix="ref_error_calc_",
                              **generic_calc_kwargs)
        calc_property_prefix = "ref_error_calc_"

    # clean up category_keys
    if isinstance(category_keys, str):
        category_keys = [category_keys]
    elif category_keys is None:
        category_keys = []

    def _reshape_normalize(diff, at):
        if per_component:
            # one long vector
            diff = diff.reshape((-1))
        elif prop in atom_properties:
            # flatten any vector/matrix dimensions so norm below is correct
            diff = diff.reshape((len(at), -1))

        if per_atom:
            diff /= len(at)

        return diff

    # compute diffs and store in all_diffs, and weights in all_weights
    all_diffs = {}
    all_parity = { "ref": {}, "calc": {} }
    all_weights = {}
    for at in inputs:
        # turn category keys into a single string for dict key
        at_category = " / ".join([str(at.info.get(k)) for k in category_keys])
        weight = at.info.get(weight_property, 1.0)

        if len(set(config_properties).intersection(set(atom_properties))) > 0:
            raise ValueError(f"Property {set(config_properties).intersection(set(atom_properties))} "
                             "appears in both config_properties and atom_properties")

        for prop in config_properties + atom_properties:
            prop_use = prop

            # parse (and remove) "/..." suffixes
            per_atom = re.search(r"/atom\b", prop_use)
            prop_use = re.sub(r"/atom\b", "", prop_use)
            per_component = re.search(r"/comp\b", prop)
            prop_use = re.sub(r"/comp\b", "", prop_use)
            by_species = re.search(r"/Z\b", prop_use)
            prop_use = re.sub(r"/Z\b", "", prop_use)

            # possibly reconstruct virial later
            virial_from_stress = False
            if prop_use == "virial":
                prop_use = "stress"
                virial_from_stress = True

            # select dict and check for inconsistencies
            if prop in config_properties:
                if by_species:
                    raise ValueError("/Z only possible in atom_properties")
                data = at.info
            else: # atom_properties
                if per_atom:
                    raise ValueError("/atom only possible in config_properties")
                data = at.arrays

            # grab data
            ref_quant = data.get(ref_property_prefix + prop_use)
            calc_quant = data.get(calc_property_prefix + prop_use)
            # skip if data is missing
            if ref_quant is None or calc_quant is None:
                # a warning here?
                continue

            if virial_from_stress:
                # ref quant was actually stress, automatically convert
                ref_quant *= -at.get_volume()
                calc_quant *= -at.get_volume()

            # make everything into an array
            if isinstance(ref_quant, (int, float)):
                ref_quant = np.asarray([ref_quant])
            if isinstance(calc_quant, (int, float)):
                calc_quant = np.asarray([calc_quant])

            if prop in config_properties:
                inds = list(range(len(ref_quant)))
                ind_groups = [(inds, "")]
            else: # atom_properties
                Zs = at.numbers
                inds = Zs
                if by_species:
                    ind_groups = [(Z, f"_{Z}") for Z in sorted(set(Zs))]
                else:
                    ind_groups = [(Zs, "")]

            for ind_val, ind_label in ind_groups:
                ref_quant =  ref_quant[inds == ind_val]
                calc_quant = calc_quant[inds == ind_val] 

                diff = calc_quant - ref_quant
                diff = _reshape_normalize(diff, at)

                # do norm of diff along all vector dimensions
                if len(diff.shape) > 1:
                    diff = np.linalg.norm(diff, axis=1)

                ref_quant = _reshape_normalize(ref_quant, at)
                calc_quant = _reshape_normalize(calc_quant, at)

                _dict_add([all_diffs, all_weights,            all_parity["ref"],   all_parity["calc"]], 
                          [diff,      _promote(weight, diff), ref_quant,           calc_quant        ],
                          at_category, prop + ind_label)

    all_diffs["_ALL_"] = all_diffs.pop("")
    all_weights["_ALL_"] = all_weights.pop("")

    all_errors = {}
    for cat in all_diffs:
        all_errors[cat] = {}
        for prop in all_diffs[cat]:
            diffs = np.asarray(all_diffs[cat][prop])
            weights = np.asarray(all_weights[cat][prop])

            RMS = np.sqrt(np.sum((diffs ** 2) * weights) / np.sum(weights))
            MAE = np.sum(np.abs(diffs) * weights) / np.sum(weights)
            num = len(diffs)

            all_errors[cat][prop] = {'RMS': RMS, 'MAE': MAE, 'num' : num}

    return all_errors, all_diffs, all_parity


def _promote(weight, val):
    try:
        return weight * np.ones(val.shape)
    except AttributeError:
        return weight


def _dict_add(dicts, values, at_category, prop):
    if at_category == "":
        cats = [at_category]
    else:
        cats = [at_category, ""]

    for d, v in zip(dicts, values):
        for cat in cats:
            if cat not in d:
                d[cat] = {}
            if prop not in d[cat]:
                d[cat][prop] = []
            d[cat][prop].extend(v)
