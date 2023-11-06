import numpy as np
from ase import data
from ase.atoms import Atoms


def prepare(outputs, atomic_numbers, bond_lengths=None, dimer_n_steps=41, dimer_factor_range=(0.65, 2.5),
            dimer_box_scale=6.0, extra_info=None, do_isolated_atoms=True, max_cutoff=None, fixed_cell=False):
    """Prepare dimer and single atom configurations

    Parameters
    ----------
    outputs: OutputSpec
        target for atom and dimer configurations
    atomic_numbers: list(int)
        list of atomic numbers
    bond_lengths : dict, None, default={Any -> 1.0}
        bond lengths by species, default is to assign 1.0 to all elements
    dimer_n_steps : int
        number of dimer steps to take
    dimer_factor_range : tuple,  default=(0.65, 2.5)
        Range of factors of bond length to use for dimer separations. If bond_length is not given, this becomes
        the exact range of distance for all pairs.
    dimer_box_scale: float, default=6.0
        dimer box size, as scale of bond length
    extra_info: dict
        extra dict to fill into atoms.info
    do_isolated_atoms: bool, default=True
        create isolated atoms in boxes as well, cell is bond_len * dimer_box_scale in all three directions or fixed cell
        requires max_cutoff
    max_cutoff: float, default None
        max cutoff, needed to ensure that isolate_atom boxes are large enough for gap_fit to accept
    fixed_cell : ase.cell.Cell, list, bool, default=False
        switch to use a fixed cell, use the cell given. Use anything that ase.cell.Cell understands

    Returns
    -------
    ConfigSet in with configs
    """
    if extra_info is None:
        extra_info = {}

    if bond_lengths is None:
        # makes dimer_factor_range distance range
        bond_lengths = {z: 1.0 for z in atomic_numbers}

    def _make_cell(bl, min_size=0.0):
        # cell from bond length or fixed one
        if fixed_cell:
            return fixed_cell
        else:
            return [max(min_size, bl * dimer_box_scale)] * 3

    if do_isolated_atoms:
        if max_cutoff is not None:
            max_cutoff += 0.1
        for z in atomic_numbers:
            at = Atoms(numbers=[z], cell=_make_cell(bond_lengths[z], max_cutoff), pbc=[False] * 3)
            at.info['config_type'] = 'isolated_atom'
            at.info.update(extra_info)
            outputs.store(at)

    for z1 in atomic_numbers:
        for z2 in atomic_numbers:
            if z1 > z2:
                continue
            bond_len_base = 0.5 * (bond_lengths[z1] + bond_lengths[z2])
            at = Atoms(numbers=[z1, z2], cell=_make_cell(bond_len_base), pbc=[False] * 3)

            for dimer_separation in np.linspace(bond_len_base * dimer_factor_range[0],
                                                bond_len_base * dimer_factor_range[1], dimer_n_steps):
                at.positions[1, 0] = dimer_separation
                at.info['config_type'] = 'dimer'
                at.info.update(extra_info)
                outputs.store(at)

    outputs.close()
    return outputs.to_ConfigSet()


def isolated_atom_from_e0(outputs, e0_dict, cell_size, energy_key="energy", extra_info=None):
    """Write single atoms with energy from e0 dictionary

    Parameters
    ----------
    outputs : Configset_out
    e0_dict : dict
    cell_size : float
        unit cell size, should be greater than cutoff
    energy_key : dict_key, default 'energy'
    extra_info: dict
        extra dict to fill into atoms.info

    Returns
    -------

    """
    if extra_info is None:
        extra_info = dict()
    if energy_key in extra_info.keys():
        raise ValueError("Energy key given in the extra info for isolated atom configuration, this is defeating the"
                         "purpose. extra_info given:", extra_info)

    for key, energy in e0_dict.items():
        if isinstance(key, int):
            key = data.chemical_symbols[key]

        at = Atoms(key, cell=[cell_size] * 3, pbc=False)
        at.info[energy_key] = energy
        at.info['config_type'] = 'isolated_atom'
        at.info.update(extra_info)
        outputs.store(at)

    outputs.close()
