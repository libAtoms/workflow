import numpy as np
import scipy.spatial

import spglib
from ase.atoms import Atom, Atoms


def find_voids(at, transl_symprec=1.0e-1, symprec=1.0e-2):
    # save original cell
    cell_orig = at.get_cell()
    reciprocal_cell_orig = at.get_reciprocal_cell()

    # create supercell
    at_sc = at * [3, 3, 3]
    at_sc.set_positions(at_sc.get_positions() - np.sum(cell_orig, axis=0))

    # calculate Voronoi tesselation
    vor = scipy.spatial.Voronoi(at_sc.get_positions())

    # list possible centers from Voronoi vertices that are close to original cell
    possible_centers_lat = np.matmul(vor.vertices, reciprocal_cell_orig.T)
    possible_indices = np.where(np.all(np.abs(possible_centers_lat - 0.5) <= 0.6, axis=1))[0]

    # create atoms object with supercell of all possible interstitial positions
    vertices = vor.vertices[possible_indices]
    at_w_interst = at.copy()
    at_w_interst.extend(Atoms('X{}'.format(len(possible_indices)), positions=vertices))

    # eliminate duplicates that are equivalent by translation
    dists = at_w_interst.get_all_distances(mic=True)
    del_list = set()
    for i in range(len(at_w_interst) - 1):
        dups = i + 1 + np.where(dists[i][i + 1:] < transl_symprec)[0]
        del_list = del_list.union(set(dups))

    del at_w_interst[list(del_list)]

    # handle symmetry
    dataset = spglib.get_symmetry_dataset((at_w_interst.cell, at_w_interst.get_scaled_positions(), at_w_interst.numbers), symprec)
    if dataset is not None:
        equivalent_indices = set(dataset["equivalent_atoms"][len(at):])
    else:
        equivalent_indices = set(range(len(at) + 1, len(at_w_interst)))

    pos = at_w_interst.get_positions()
    voids = []
    for i in equivalent_indices:
        at_t = at + Atom('H')
        p = at_t.get_positions()
        p[-1] = pos[i]
        at_t.set_positions(p)
        d = min(at_t.get_distances(len(at_t) - 1, range(len(at_t) - 1), mic=True))
        voids.append((d, pos[i][0], pos[i][1], pos[i][2]))

    return sorted(voids, key=lambda x: x[0], reverse=True)
