import warnings

import numpy as np
from ase import neighborlist
from wfl.utils.misc import atoms_to_list

from wfl.generate.utils import config_type_append


def abstract_sp3_hydrogen_atoms(input_mol, label_config_type=True,
                                cutoffs=None):
    """ Removes molecule's sp3 hydrogen atoms one at a time to give a number
    of corresponding unsaturated (radical) structures.

    Method of determining sp3: carbon has 4 neighbors of any kind. Only
    removes from Carbon atoms.

    Parameters
    ----------

    inputs: Atoms
        structure to remove sp3 hydrogen atoms from
    label_config_type: bool, default True
        whether to append config_type with 'mol' or 'rad{idx}'
    cutoffs: dict, default None
        cutoffs to override default values from neighborlist.natural_cutoffs()

    Returns
    -------
        list(Atoms): All sp3 radicals corresponding to input molecule

    """

    natural_cutoffs = neighborlist.natural_cutoffs(input_mol)

    if cutoffs is not None:
        natural_cutoffs.update(cutoffs)

    neighbor_list = neighborlist.NeighborList(natural_cutoffs,
                                              self_interaction=False,
                                              bothways=True)
    _ = neighbor_list.update(input_mol)

    symbols = np.array(input_mol.symbols)
    sp3_hs = []
    for at in input_mol:
        if at.symbol == 'H':
            h_idx = at.index

            indices, offsets = neighbor_list.get_neighbors(h_idx)
            if len(indices) != 1:
                raise RuntimeError("Got no or more than one hydrogen "
                                   "neighbors")

            # find index of the atom H is bound to
            h_neighbor_idx = indices[0]

            if symbols[h_neighbor_idx] != 'C':
                continue

            # count number of neighbours of the carbon H is bound to
            indices, offsets = neighbor_list.get_neighbors(h_neighbor_idx)

            no_carbon_neighbors = len(indices)

            if no_carbon_neighbors == 4:
                sp3_hs.append(h_idx)

    if len(sp3_hs) == 0:
        warnings.warn("No sp3 hydrogens were found; no radicals returned")
        return []

    radicals = []
    for h_idx in sp3_hs:
        at = input_mol.copy()
        del at[h_idx]
        radicals.append(at)

    if label_config_type:
        for rad, h_id in zip(radicals, sp3_hs):
            config_type_append(rad, f'rad{h_id}')

    return radicals

