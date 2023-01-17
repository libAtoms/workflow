from collections import Counter
import numpy as np
import warnings
from wfl.configset import ConfigSet, OutputSpec


def find_isolated_atoms(inputs, outputs, isolated_atom_info_key="config_type", 
    isolated_atom_info_value="default"):
    """Finds isolated atoms in among all configs. 
    
    Parameters
    ----------
    inputs: list[Atoms] or ConfigSet
        all configs to search through
    outputs: OutputSpec 
        where to save isolated atoms to
    isolated_atom_info_key: str, default "config_type"
        key for Atoms.info to select isolated atoms by
    isolated_atom_info_value: str, default "default"
        value of Atoms.info[isolated_atom_info_key] to match isolated atoms on
        "default" matches "isolated_atom" or "IsolatedAtom".
    """

    if outputs.all_written():
        warnings.warn(f"Not searching for isolated atoms because {outputs} is done.")
        return outputs.to_ConfigSet()

    if isolated_atom_info_value == "default":
        isolated_atom_info_value = ["isolated_atom", "IsolatedAtom"]
    elif isinstance(isolated_atom_info_value, str):
        isolated_atom_info_value = [isolated_atom_info_value]

    isolated_atoms = []
    for at in inputs:
        if isolated_atom_info_key in at.info and at.info[isolated_atom_info_key] in isolated_atom_info_value:
            if len(at) != 1:
                raise RuntimeError(f'Config marked as an isolated atom, but has more than one atom in `Atoms`.')
            isolated_atoms.append(at)

    found_symbols = [str(at.symbols) for at in isolated_atoms] 
    for symbol, count in Counter(found_symbols).items():
        if count != 1:
            raise RuntimeError(f"Isolated atom for element {symbol} found more than once ({count} times).")

    outputs.store(isolated_atoms)
    outputs.close()
    return outputs.to_ConfigSet()


def atomization_energy(inputs, outputs, prop_prefix, property="energy", isolated_atom_info_key="config_type", 
    isolated_atom_info_value="default"):
    """ Calculates atomization energy. 
    
    Parameters
    ----------
    inputs: list[Atoms] or ConfigSet
        all configs, including isolated atoms
    outputs: OutputSpec 
        for saving structures with atomization energies 
    prop_prefix: str
        prefix for reading total energy (e.g. Atoms.info[f"{prop_prefix}energy"])
        and saving atomization energy (Atoms.info[f"{prop_prefix}atomization_energy"]) 
    property: str, default "energy"
        name for property to read from Atoms.info (Atoms.info["{prop_prefix}{property}])
    isolated_atom_info_key: str, default "config_type"
        key for Atoms.info to select isolated atoms by
    isolated_atom_info_value: str, default "default"
        value of Atoms.info[isolated_atom_info_key] to match isolated atoms on
        "default" matches "isolated_atom" or "IsolatedAtom".
    """

    if outputs.is_done():
        warnings.warn(f"Ouput for atomization energy ({outputs}) are done, not recomputing.")
        return outputs.to_ConfigSet()

    isolated_atoms = find_isolated_atoms(
        inputs=inputs,
        outputs=OutputSpec(),
        isolated_atom_info_key=isolated_atom_info_key,
        isolated_atom_info_value=isolated_atom_info_value
        )

    isolated_at_data = {}
    for at in isolated_atoms:
        isolated_at_data[list(at.symbols)[0]] = at.info[f'{prop_prefix}{property}']

    for at in inputs:
        counted_ats = Counter(list(at.symbols))
        assert counted_ats.keys() == isolated_at_data.keys(), f"have isolated atom energies for {isolated_at_data.keys()}, but config has {counted_ats.keys()} elements."
        binding_energy = at.info[f'{prop_prefix}{property}']
        for symbol, count in counted_ats.items():
            binding_energy -= count * isolated_at_data[symbol]

        at.info[f"{prop_prefix}atomization_{property}"] = -1 * binding_energy 
        outputs.store(at)

    outputs.close()
    return outputs.to_ConfigSet()

