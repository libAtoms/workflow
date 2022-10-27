import warnings
import re

import numpy as np

from wfl.configset import ConfigSet, OutputSpec
from wfl.calculators.generic import run as generic_calc


def calc(inputs, calc_property_prefix, ref_property_prefix,
         config_properties=None, atom_properties=None, category_keys="config_type",
         weight_property=None, missing_property_error=False,
         calc_outputs=None, calc_autopara_info=None):
    """calculate error for calculation results relative to stored reference values

    Parameters
    ----------
    inputs: iterable(Atoms)
        input configurations with reference and calculated values in Atoms.info/arrays
    calc_property_prefix: str
        prefix to info/array keys for calculated properties
    ref_property_prefix: str
        prefix to info/array keys for reference properties
    config_properties: list(str), default ["energy/atom", "virial/atom/comp"]
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
    missing_property_error: bool, default False
        raise a ValueError if a property that's requested is missing
    calc_outputs: OutputSpec, optional
        where to store configs with calculated properties from optional calculation of
        properties to be tested
    calc_autopara_info: AutoparaInfo, optional
        autoparallelization info for optional initial calculation of properties to be tested

    Returns
    -------
        errors: dict of RMS and MAE errors for each category and property
        diffs: dict with list of differences for each category and property
        parity: dict with "ref" and "calc" keys, each containing list of property values for
            each category and property, for parity plots
    """

    # default properties
    if config_properties is None and atom_properties is None:
        config_properties = ["energy/atom", "virial/atom/comp"]
        atom_properties = ["forces"]
    if config_properties is None:
        config_properties = []
    if atom_properties is None:
        atom_properties = []

    # clean up category_keys
    if isinstance(category_keys, str):
        category_keys = [category_keys]
    elif category_keys is None:
        category_keys = []

    def _reshape_normalize(quant, prop, atoms, per_atom):
        """reshape and normalize quantity so its error can be calculated cleanly

        Parameters
        ----------
        quant: scalar number or array of numbers
            quantity whose error will be computed.  Can be scalar or arbitrary shape
            array (if per-config) or array with leading dimension len(at) (if per-atom)
        prop: str
            name of property
        atoms: Atoms
            corresponding atoms object
        per_atom: bool
            quantity should be divided by len(atoms)

        Returns
        -------
        quant: 2-d array containing reshaped quantity, with leading dimension 1 for per-config
            or len(atoms) for per-atom
        """
        # convert scalars or lists into arrays
        quant = np.asarray(quant)

        # Reshape to 2-d, with leading dimension 1 for per-config, and len(atoms) for per-atom.
        # This is the right shape to work with later flattening for per-property and norm calculation
        # for vector property differences.
        if prop in config_properties:
            quant = quant.reshape((1, -1))
        else:
            # flatten all except per-atom dimension
            quant = quant.reshape((len(atoms), -1))

        if per_atom:
            quant /= len(atoms)

        return quant

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
                if missing_property_error:
                    raise ValueError(f"Required ref_quant {ref_property_prefix + prop_use} is None {ref_quant is None} "
                                     f"or calc_quant {calc_property_prefix + prop_use} is None {calc_quant is None}")
                continue

            if virial_from_stress:
                # ref quant was actually stress, automatically convert
                ref_quant *= -at.get_volume()
                calc_quant *= -at.get_volume()

            # make everything into an appropriately shaped and normalized array
            ref_quant = _reshape_normalize(ref_quant, prop, at, per_atom)
            calc_quant = _reshape_normalize(calc_quant, prop, at, per_atom)

            if prop in config_properties or not by_species:
                # If quantities are not being split up by species, make a
                # group that includes all atoms, so loop below that separates out
                # things by Z will lump them all together.
                atom_split_indices = np.asarray(range(len(ref_quant)))
                atom_split_groups = [(atom_split_indices, "")]
            else: # atom_properties
                # Make separate groups for each atomic number Z, so their errors
                # can be tabulated separately.
                Zs = at.numbers
                atom_split_indices = np.asarray(Zs)
                atom_split_groups = [(Z, f"_{Z}") for Z in sorted(set(Zs))]

            # Loop over groups that errors should be split up by within each configuration.
            # Only atomic number Z implemented so far (see above).
            for atom_split_index_val, atom_split_index_label in atom_split_groups:
                # use only subset of quantities that are relevant to this subset of atoms,
                # normally either all atoms or the ones with one particular atomic number
                ref_quant = ref_quant[atom_split_indices == atom_split_index_val]
                calc_quant = calc_quant[atom_split_indices == atom_split_index_val]

                if per_component:
                    # if per component, flatten all vector component so each is counted separately
                    ref_quant = ref_quant.reshape((-1, 1))
                    calc_quant = calc_quant.reshape((-1, 1))

                diff = calc_quant - ref_quant

                if len(diff.shape) != 2:
                    raise RuntimeError(f"Should never have diff.shape={diff.shape} with dim != 2 (prop {prop + atom_split_index_label})")
                # compute norm along vector components
                diff = np.linalg.norm(diff, axis=1)

                _dict_add([all_diffs, all_weights,            all_parity["ref"],   all_parity["calc"]], 
                          [diff,      _promote(weight, diff), ref_quant,           calc_quant        ],
                          at_category, prop + atom_split_index_label)

    all_diffs["_ALL_"] = all_diffs.pop("")
    all_weights["_ALL_"] = all_weights.pop("")
    for k in all_parity.keys():
        all_parity[k]["_ALL_"] = all_parity[k].pop("")

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
