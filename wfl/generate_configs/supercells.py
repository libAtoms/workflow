import math

import ase.build
import numpy as np
import spglib
from ase.atoms import Atoms

from wfl.pipeline import iterable_loop
from wfl.utils.find_voids import find_voids


def largest_bulk(inputs, outputs, max_n_atoms, primitive=True, symprec=1.0e-3, chunksize=10):
    return iterable_loop(iterable=inputs, configset_out=outputs, op=largest_bulk_op,
                         chunksize=chunksize, max_n_atoms=max_n_atoms, primitive=primitive, symprec=symprec)


def vacancy(inputs, outputs, max_n_atoms, primitive=True, symprec=1.0e-3, chunksize=10):
    return iterable_loop(iterable=inputs, configset_out=outputs, op=vacancy_op,
                         chunksize=chunksize, max_n_atoms=max_n_atoms, primitive=primitive, symprec=symprec)


def interstitial(inputs, outputs, max_n_atoms, interstitial_probability_radius_exponent=3.0,
                 primitive=True, symprec=1.0e-3, chunksize=10):
    return iterable_loop(iterable=inputs, configset_out=outputs, op=interstitial_op,
                         chunksize=chunksize, max_n_atoms=max_n_atoms,
                         interstitial_probability_radius_exponent=interstitial_probability_radius_exponent,
                         primitive=primitive, symprec=symprec)


def surface(inputs, outputs, max_n_atoms, min_thickness, vacuum,
            simple_cut=False, max_surface_cell_indices=1, duplicate_in_plane=True, pert=0.0,
            primitive=True, symprec=1.0e-3, chunksize=10):
    return iterable_loop(iterable=inputs, configset_out=outputs, op=surface_op,
                         chunksize=chunksize, max_n_atoms=max_n_atoms,
                         min_thickness=min_thickness, vacuum=vacuum, simple_cut=simple_cut,
                         max_surface_cell_indices=max_surface_cell_indices,
                         duplicate_in_plane=duplicate_in_plane, pert=pert,
                         primitive=primitive, symprec=symprec)


def _get_primitive(at, symprec=1.0e-3):
    """get primitive cell corresponding to original lattice, copying original cell info and
    storing primitive cell in info

    Parameters
    ----------
    at: Atoms
        original atomic structure
    symprec: float, default 1e-3
        precision for spglib symmetry calculation

    Returns
    -------
    Atoms with primitive cell, info from original Atoms object, and new info field 'primitive_cell'
    """

    lat_pos_nums = spglib.standardize_cell(at, to_primitive=True, symprec=symprec)
    if lat_pos_nums is not None:
        # successfully identified primitive cell
        prim_at = Atoms(numbers=lat_pos_nums[2], cell=lat_pos_nums[0], scaled_positions=lat_pos_nums[1], pbc=[True] * 3)
        prim_at.info.update(at.info)
        prim_at.info['primitive_cell'] = np.array(prim_at.cell)
        at = prim_at
    return at


# NOTE: currently only makes supercell with diagonal transformation matrix,
# i.e. multiplying each cell vector by an integer.  Should this be generalized
# to linear combination to make more optimal (cube-like, largest possible) cells?
def _largest_isotropic_supercell(at, max_n_atoms, vary_cell_vectors=None):
    """Calculate duplicates for largest supercell consistent with max n atoms,
    trying to make duplicated cell vectors equal length

    Parameters
    ----------
    at: Atoms
        original atoms object
    max_n_atoms: int
        max number of atoms allowed in supercell
    vary_cell_vectors: list(int)
        list of indices of lattice vectors that can be varied

    Returns
    -------
    list(int) with factor for each cell vector
    """

    if vary_cell_vectors is None:
        vary_cell_vectors = [0, 1, 2]
    orig_cell = at.get_cell()
    n_dups = [1] * 3
    while True:
        t_cell = (orig_cell.T * n_dups).T
        min_cell_i = np.argmin(np.linalg.norm(t_cell[vary_cell_vectors], axis=1))
        min_cell_i = vary_cell_vectors[min_cell_i]
        n_dups[min_cell_i] += 1
        if np.product(n_dups) * len(at) > max_n_atoms:
            n_dups[min_cell_i] -= 1
            break
    return n_dups


# try ase.build.supercells.find_optimal_cell_shape ?
def largest_bulk_op(atoms, max_n_atoms, pert=0.0, primitive=True, symprec=1.0e-3):
    """make largest bulk-like supercells
    Parameters
    ----------
    atoms: list(Atoms)
        input primitive cell configs
    max_n_atoms: int
        maximum number of atoms allowed in a supercell
    pert: float, default 0.0
        magnitude of random displacement
    primitive: bool, default True
        reduce input Atoms to primitive cell using spglib
    symprec: float, default 1.0e-3
        symprec for primitive lattice check

    Returns
    -------
        list(Atoms) supercells
    """
    if isinstance(atoms, Atoms):
        atoms = [atoms]

    supercells = []
    for at in atoms:
        if primitive:
            at = _get_primitive(at, symprec=symprec)

        n_dups = _largest_isotropic_supercell(at, max_n_atoms)

        at.info['orig_cell'] = np.array(at.cell)
        at *= n_dups

        at.info["supercell_n"] = n_dups
        at.info["config_type"] = "supercell_bulk"

        at.positions += pert * np.random.normal(size=at.positions.shape)
        ####################################################################################################
        # workaround for non-working results invalidation when number of atoms has changed
        at.calc = None
        ####################################################################################################

        supercells.append(at)

    return supercells


def vacancy_op(atoms, max_n_atoms, pert=0.0, primitive=True, symprec=1.0e-3):
    """make vacancies in largest bulk-like supercells
    Parameters
    ----------
    atoms: list(Atoms)
        input primitive cell configs
    max_n_atoms: int
        maximum number of atoms allowed in a supercell
    pert: float, default 0.0
        magnitude of random displacement
    primitive: bool, default True
        reduce input Atoms to primitive cell using spglib
    symprec: float, default 1.0e-3
        symprec for primitive lattice check

    Returns
    -------
        list(Atoms) supercells with vacancies
    """
    supercells = largest_bulk_op(atoms, max_n_atoms, primitive=primitive, symprec=symprec)
    for at in supercells:
        vac_i = np.random.choice(len(at))
        at.info["vacancy_Z"] = at.numbers[vac_i]
        at.info["vacancy_pos"] = at.positions[vac_i]
        del at[vac_i]
        at.info["config_type"] = "supercell_vacancy"

        at.positions += pert * np.random.normal(size=at.positions.shape)
        ####################################################################################################
        # workaround for non-working results invalidation when number of atoms has changed
        at.calc = None
        ####################################################################################################
    return supercells


def interstitial_op(atoms, max_n_atoms, pert=0.0, interstitial_probability_radius_exponent=3.0, primitive=True,
                    symprec=1.0e-3):
    """make interstitials in largest bulk-like supercells
    Parameters
    ----------
    atoms: list(Atoms)
        input primitive cell configs
    max_n_atoms: int
        maximum number of atoms allowed in a supercell
    pert: float, default 0.0
        magnitude of random displacement
    interstitial_probability_radius_exponent: float, default 3.0
        probability of selecting an interstitial void is proportional to its radius to this power
    primitive: bool, default True
        reduce input Atoms to primitive cell using spglib
    symprec: float, default 1.0e-3
        symprec for primitive lattice check

    Returns
    -------
        list(Atoms) supercells with interstitials
    """
    supercells = largest_bulk_op(atoms, max_n_atoms, primitive=primitive, symprec=symprec)
    for at in supercells:
        voids = np.asarray(find_voids(at))
        p_raw = voids[:, 0] ** interstitial_probability_radius_exponent
        i_ind = np.random.choice(range(len(p_raw)), p=p_raw / np.sum(p_raw))
        i_pos = voids[i_ind][1:4]
        # NOTE: probably need a better heuristic for multicomponent systems
        #       does it need to be as detailed as a per-group probability distribution?
        i_Z = np.random.choice(list(set(at.numbers)))
        at += Atoms(numbers=[i_Z], positions=[i_pos])
        at.info["interstitial"] = len(at) - 1
        at.info["config_type"] = "supercell_interstitial"

        at.positions += pert * np.random.normal(size=at.positions.shape)
        ####################################################################################################
        # workaround for non-working results invalidation when number of atoms has changed
        at.calc = None
        ####################################################################################################
    return supercells


def _are_lin_dep(v1, v2):
    return np.abs(np.abs(np.dot(v1, v2)) - np.linalg.norm(v1) * np.linalg.norm(v2)) < 1.0e-10


def surface_op(atoms, max_n_atoms, min_thickness, vacuum, simple_cut=False, max_surface_cell_indices=1,
               duplicate_in_plane=True, pert=0.0,
               primitive=True, symprec=1.0e-3):
    """make surface supercells
    Parameters
    ----------
    atoms: list(Atoms)
        input primitive cell configs
    max_n_atoms: int
        maximum number of atoms allowed in a supercell
    min_thickness: float
        duplicate slab in normal direction to exceed this minimum thickness
    vacuum: float
        thickness of vacuum layer to add
    simple_cut: bool, default False
        if true, surface is formed by two primitive lattice vectors, otherwise use two random
        vectors with indices up to +/- max_surface_cell_indices
    max_surface_cell_indices: int, default 1
        maximum indices used to make surfaces when simple_cut=False
    duplicate_in_plane: bool, default True
        duplicate surface cell in-plane if allowed by max_n_atoms
    pert: float, default 0.0
        magnitude of random displacement
    primitive: bool, default True
        reduce input Atoms to primitive cell using spglib
    symprec: float, default 1.0e-3
        symprec for primitive lattice check

    Returns
    -------
        list(Atoms) surface supercells with surface formed by first two cell vectors
    """

    supercells = []
    for at in atoms:
        if primitive:
            at = _get_primitive(at, symprec=symprec)

        if simple_cut:
            # surface plane formed by two lattice vectors

            # out of plane vector random or longest better?
            # surf_i = np.random.randint(3)
            surf_i = np.argmax(np.linalg.norm(at.get_cell(), axis=1))

            other_indices = [0, 1, 2]
            del other_indices[surf_i]

            # create transformation matrix
            s0 = np.zeros(3, dtype=int)
            s1 = np.zeros(3, dtype=int)
            s2 = np.zeros(3, dtype=int)
            s0[other_indices[0]] = 1
            s1[other_indices[1]] = 1
            s2[surf_i] = 1

        else:
            # surface plane formed by 2 supercell vectors up to max_surface_cell_indices
            # pick 1st vector
            s0 = np.zeros(3, dtype=int)
            while np.sum(np.abs(s0)) == 0:
                s0 = np.random.choice(range(-max_surface_cell_indices, max_surface_cell_indices + 1), 3)
            # pick 2nd vector that is not linearly dependent
            s1 = np.zeros(3, dtype=int)
            while np.sum(np.abs(s1)) == 0 or _are_lin_dep(s0, s1):
                s1 = np.random.choice(range(-max_surface_cell_indices, max_surface_cell_indices + 1), 3)

            # pick 3rd vector that is not in plane
            s2 = np.zeros(3, dtype=int)
            for i2 in range(3):
                s2[i2] = 1
                if np.abs(np.dot(s2, np.cross(s0, s1))) > 1.0e-6:
                    break

        # create supercell
        transformation = np.asarray([s0, s1, s2])
        if np.linalg.det(transformation) < 0:
            (transformation[0], transformation[1]) = (transformation[1].copy(), transformation[0].copy())
        sc = ase.build.make_supercell(at, transformation)
        sc.info.update(at.info)
        at = sc

        # duplicate along out-of-plane direction to reach min_thickness
        c = at.get_cell()
        surf_norm_hat = np.cross(c[0], c[1])
        surf_norm_hat /= np.linalg.norm(surf_norm_hat)
        slab_thickness = np.abs(np.dot(c[2], surf_norm_hat))
        thickness_n_dup = np.floor(min_thickness / slab_thickness) + 1
        size_n_dup = max(np.floor(max_n_atoms / len(at)), 1)

        n_dup = np.asarray([1] * 3)
        n_dup[2] = min(thickness_n_dup, size_n_dup)
        at *= n_dup

        # duplicate in plane for more variability if possible
        if duplicate_in_plane and len(at) < max_n_atoms:
            # duplicate only in surface cell vectors 0,1
            n_dup = _largest_isotropic_supercell(at, max_n_atoms, [0, 1])
            at *= n_dup

        # store final transformation
        transformation = np.multiply(transformation.T, n_dup).T
        at.info['transformation'] = transformation

        if vacuum > 0:
            # add vacuum layer and make 3rd vector normal to surface
            c = at.get_cell()
            surf_norm_hat = np.cross(c[0], c[1])
            surf_norm_hat /= np.linalg.norm(surf_norm_hat)
            dot_prod = np.dot(c[2], surf_norm_hat)

            c[2] = surf_norm_hat * (dot_prod + math.copysign(vacuum, dot_prod))
            at.set_cell(c, False)
            at.wrap()

            at.info["config_type"] = "supercell_surface"
        else:
            at.info["config_type"] = "supercell_slab"

        at.positions += pert * np.random.normal(size=at.positions.shape)
        ####################################################################################################
        # workaround for non-working results invalidation when number of atoms has changed
        at.calc = None
        ####################################################################################################

        supercells.append(at)

    return supercells
