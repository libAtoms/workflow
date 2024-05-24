"""
Committee of Models

Calculated properties with a list of models and saves them into info/arrays.
Further operations (eg. mean, variance, etc.) with these are up to the user.
"""
from ase import Atoms

from wfl.utils.save_calc_results import per_atom_properties, per_config_properties
from wfl.utils.misc import atoms_to_list
from wfl.utils.parallel import construct_calculator_picklesafe

__default_properties = ['energy', 'forces', 'stress']


def calculate_committee(atoms, calculator_list, properties=None, output_prefix="committee_{}_"):
    """Calculate energy and forces with a committee of models

    Notes
    -----
    Supports formatter string in the output_prefix arg, but currently only with a single field literally "{}".

    Parameters
    ----------
    atoms : Atoms / list(Atoms)
        input atomic configs
    calculator_list : list(Calculator) / list[(initializer, args, kwargs)]
        list of calculators to use as a committee of models on the configs
    properties: list[str], default ['energy', 'forces', 'stress']
        properties to calculate
    output_prefix : str, default="committee\_"
        prefix for results coming from the committee of models.
        If includes "{}" then will use it as a format string, otherwise puts a number at the end of prefix for the
        index of the model in the committee of models

    Returns
    -------
    atoms : Atoms / list(Atoms)

    """
    if properties is None:
        properties = __default_properties

    atoms_list = atoms_to_list(atoms)

    # create the general key formatter
    if "{}" in output_prefix:
        if output_prefix.count("{}") != 1:
            raise ValueError("Prefix with formatting is incorrect, cannot have more than one of `{}` in it")
        key_formatter = f"{output_prefix}{{}}"
    else:
        key_formatter = f"{output_prefix}{{}}{{}}"

    # create calculator instances
    calculator_list_to_use = [construct_calculator_picklesafe(calc) for calc in calculator_list]

    for at in atoms_list:
        # calculate forces and energy with all models from the committee
        for i_model, pot in enumerate(calculator_list_to_use):
            for prop in properties:
                if prop in per_atom_properties:
                    at.arrays[key_formatter.format(i_model, prop)] = pot.get_property(name=prop, atoms=at)
                elif prop in per_config_properties:
                    at.info[key_formatter.format(i_model, prop)] = pot.get_property(name=prop, atoms=at)
                else:
                    raise ValueError("Don't know where to put property: {}".format(prop))

    if isinstance(atoms, Atoms):
        return atoms_list[0]
    else:
        return atoms_list
