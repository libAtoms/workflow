import math
import warnings

import ase.build
import numpy as np
import spglib
from ase.atoms import Atoms

from wfl.autoparallelize import autoparallelize, autoparallelize_docstring
from wfl.utils.find_voids import find_voids

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

    lat_pos_nums = spglib.standardize_cell((at.cell, at.get_scaled_positions(), at.numbers), to_primitive=True, symprec=symprec)
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
def _largest_bulk_autopara_wrappable(atoms, max_n_atoms, pert=0.0, primitive=True, symprec=1.0e-3, ase_optimal=False,
                                     rng=None, _autopara_per_item_info=None):
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
    ase_optimal: bool, default False
        use ase.build.supercells.find_optimal_cell_shape
    rng: numpy.random.Generator
        random number generator
    _autopara_per_item_info: list
        internal use

    Returns
    -------
        list(Atoms) supercells
    """
    if isinstance(atoms, Atoms):
        atoms = [atoms]

    supercells = []
    for at_i, at in enumerate(atoms):
        if primitive:
            at = _get_primitive(at, symprec=symprec)

        at.info['orig_cell'] = np.array(at.cell)

        if ase_optimal:
            n_dups = ase.build.supercells.find_optimal_cell_shape(at.cell, max_n_atoms // len(at), 'sc')  # , lower_limit=-4, upper_limit=4)
            sc = ase.build.make_supercell(at, n_dups)
            sc.info.update(at.info)
            at = sc
        else:
            n_dups = _largest_isotropic_supercell(at, max_n_atoms)
            at *= n_dups

        at.info["supercell_n"] = n_dups

        at.info["config_type"] = "supercell_bulk"

        if pert != 0.0:
            at.positions += pert * _autopara_per_item_info[at_i]["rng"].normal(size=at.positions.shape)

        ####################################################################################################
        # workaround for non-working results invalidation when number of atoms has changed
        at.calc = None
        ####################################################################################################

        supercells.append(at)

    return supercells


def largest_bulk(*args, **kwargs):
    return autoparallelize(_largest_bulk_autopara_wrappable, *args,
                           default_autopara_info={"num_inputs_per_python_subprocess": 10}, **kwargs)
autoparallelize_docstring(largest_bulk, _largest_bulk_autopara_wrappable, "Atoms")


def _vacancy_autopara_wrappable(atoms, max_n_atoms, pert=0.0, primitive=True, symprec=1.0e-3, n_vac=1, cluster_r=0.0,
                                rng=None, _autopara_per_item_info=None):
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
    n_vac: int, default 1
        number of vacancies to create in each supercell
    cluster_r: float, default 0.0
        if > 0.0, multiply by 1st nn distance of initial vacancy, and make vacancies in cluster of
        atoms within this distance of initial vacancy.
    rng: numpy.random.Generator
        random number generator
    _autopara_per_item_info: list
        internal use

    Returns
    -------
        list(Atoms) supercells with vacancies
    """
    supercells = _largest_bulk_autopara_wrappable(atoms, max_n_atoms, primitive=primitive, symprec=symprec)
    for at_i, at in enumerate(supercells):
        if len(at) <= n_vac:
            # should this be an error?
            warnings.warn(f'Cannot make {n_vac} vacancies in structure with {len(at)} atoms')
            continue

        rng = _autopara_per_item_info[at_i]['rng']

        if cluster_r > 0.0:
            # cluster
            nns = []
            n_try = 0
            while n_try < 20:
                init_vac_i = rng.choice(len(at))
                inds = list(range(len(at)))
                dists = at.get_distances(init_vac_i, inds, mic=True)
                nearest_d = np.amin(dists[np.where(dists > 0.0)])
                nns = np.where(np.logical_and(dists <= cluster_r * nearest_d, dists > 0.0))[0]
                if len(nns) >= n_vac:
                    break
                n_try += 1

            if len(nns) < n_vac:
                # should this be an error?
                warnings.warn(f'Failed to find vacancy with at least {n_vac}-1 neighbors after 20 tries')
                continue

            vac_i = np.concatenate(([init_vac_i], rng.choice(nns, size=n_vac - 1, replace=False)))
        else:
            # entirely random set
            vac_i = rng.choice(len(at), size=n_vac, replace=False)

        at.info["vacancy_Z"] = at.numbers[vac_i]
        at.info["vacancy_pos"] = at.positions[vac_i]
        del at[vac_i]
        at.info["config_type"] = "supercell_vacancy"
        if n_vac > 1:
            at.info["config_type"] += f'_{n_vac}'
        if cluster_r > 0.0:
            at.info["config_type"] += f'_r_{cluster_r}_{nearest_d:.3f}'


        if pert != 0.0:
            at.positions += pert * rng.normal(size=at.positions.shape)
        ####################################################################################################
        # workaround for non-working results invalidation when number of atoms has changed
        at.calc = None
        ####################################################################################################
    return supercells


def vacancy(*args, **kwargs):
    return autoparallelize(_vacancy_autopara_wrappable, *args, default_autopara_info={"num_inputs_per_python_subprocess": 10}, **kwargs)
autoparallelize_docstring(vacancy, _vacancy_autopara_wrappable, "Atoms")


def _antisite_autopara_wrappable(atoms, max_n_atoms, pert=0.0, primitive=True, symprec=1.0e-3, n_antisite=1, cluster_r=0.0, Zs=None,
                                 rng=None, _autopara_per_item_info=None):
    """make antisites in largest bulk-like supercells

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
    n_antisite: int, default 1
        number of antisites to create in each supercell
    cluster_r: float, default 0.0
        if > 0.0, multiply by 1st nn distance of initial antisite, and make antisites in cluster of
        atoms within this distance of initial antisite.
    Zs: iterable(int), default None
        list of atomic numbers to be used to create antisites.  If none or length 0, use set of species already present
        in each config for itself.
    rng: numpy.random.Generator
        random number generator
    _autopara_per_item_info: list
        internal use

    Returns
    -------
        list(Atoms) supercells with antisites
    """
    supercells = _largest_bulk_autopara_wrappable(atoms, max_n_atoms, primitive=primitive, symprec=symprec)
    for at_i, at in enumerate(supercells):
        if len(at) <= n_antisite:
            # should this be an error?
            warnings.warn(f"Cannot make {n_antisite} antisites in structure with {len(at)} atoms")
            continue

        rng = _autopara_per_item_info[at_i]['rng']

        if Zs is None or len(Zs) == 0:
            avail_Zs = set(at.numbers)
        else:
            avail_Zs = set(Zs)

        if len(set(at.numbers)) == 1 and set(at.numbers) == avail_Zs:
            warnings.warn(f"Cannot make {n_antisite} antisites in structure with only one species with only same species available")
            continue

        if cluster_r > 0.0:
            # cluster
            nns = []
            n_try = 0
            while n_try < 20:
                init_antisite_i = rng.choice(len(at))
                inds = list(range(len(at)))
                dists = at.get_distances(init_antisite_i, inds, mic=True)
                nearest_d = np.amin(dists[np.where(dists > 0.0)])
                nns = np.where(np.logical_and(dists <= cluster_r * nearest_d, dists > 0.0))[0]
                if len(nns) >= n_antisite:
                    break
                n_try += 1

            if len(nns) < n_antisite:
                # should this be an error?
                warnings.warn(f'Failed to find antisite with at least {n_antisite}-1 neighbors after 20 tries')
                continue

            antisite_i = np.concatenate(([init_antisite_i], rng.choice(nns, size=n_antisite - 1, replace=False)))
        else:
            # entirely random set
            antisite_i = rng.choice(len(at), size=n_antisite, replace=False)

        antisite_Zs = []
        for ii in antisite_i:
            Z_new = rng.choice(list(avail_Zs - set([at.numbers[ii]])))
            antisite_Zs.append([at.numbers[ii], Z_new])
            at.numbers[ii] = Z_new

        at.info["antisite_Zs"] = np.asarray(antisite_Zs)
        at.info["antisite_pos"] = at.positions[antisite_i]
        at.info["config_type"] = "supercell_antisite"
        if n_antisite > 1:
            at.info["config_type"] += f'_{n_antisite}'
        if cluster_r > 0.0:
            at.info["config_type"] += f'_r_{cluster_r}_{nearest_d:.3f}'


        if pert != 0.0:
            at.positions += pert * rng.normal(size=at.positions.shape)
        ####################################################################################################
        # workaround for non-working results invalidation when identity of atoms has changed
        at.calc = None
        ####################################################################################################
    return supercells


def antisite(*args, **kwargs):
    return autoparallelize(_antisite_autopara_wrappable, *args, default_autopara_info={"num_inputs_per_python_subprocess": 10}, **kwargs)
autoparallelize_docstring(antisite, _antisite_autopara_wrappable, "Atoms")


def _interstitial_autopara_wrappable(atoms, max_n_atoms, pert=0.0, interstitial_probability_radius_exponent=3.0, primitive=True,
                                     symprec=1.0e-3, rng=None, _autopara_per_item_info=None):
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
    rng: numpy.random.Generator
        random number generator
    _autopara_per_item_info: list
        internal use

    Returns
    -------
        list(Atoms) supercells with interstitials
    """
    supercells = _largest_bulk_autopara_wrappable(atoms, max_n_atoms, primitive=primitive, symprec=symprec)
    for at_i, at in enumerate(supercells):
        rng = _autopara_per_item_info[at_i]['rng']

        voids = np.asarray(find_voids(at))
        p_raw = voids[:, 0] ** interstitial_probability_radius_exponent
        i_ind = rng.choice(range(len(p_raw)), p=p_raw / np.sum(p_raw))
        i_pos = voids[i_ind][1:4]
        # NOTE: probably need a better heuristic for multicomponent systems
        #       does it need to be as detailed as a per-group probability distribution?
        i_Z = rng.choice(list(set(at.numbers)))
        at += Atoms(numbers=[i_Z], positions=[i_pos])
        at.info["interstitial"] = len(at) - 1
        at.info["config_type"] = "supercell_interstitial"

        if pert != 0.0:
            at.positions += pert * rng.normal(size=at.positions.shape)
        ####################################################################################################
        # workaround for non-working results invalidation when number of atoms has changed
        at.calc = None
        ####################################################################################################
    return supercells


def interstitial(*args, **kwargs):
    return autoparallelize(_interstitial_autopara_wrappable, *args,
                           default_autopara_info={"num_inputs_per_python_subprocess": 10}, **kwargs)
autoparallelize_docstring(interstitial, _interstitial_autopara_wrappable, "Atoms")


def _are_lin_dep(v1, v2):
    return np.abs(np.abs(np.dot(v1, v2)) - np.linalg.norm(v1) * np.linalg.norm(v2)) < 1.0e-10


def _surface_autopara_wrappable(atoms, max_n_atoms, min_thickness, vacuum, simple_cut=False, max_surface_cell_indices=1,
               duplicate_in_plane=True, pert=0.0, primitive=True, symprec=1.0e-3,
               rng=None, _autopara_per_item_info=None):
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
    rng: numpy.random.Generator
        random number generator
    _autopara_per_item_info: list
        internal use

    Returns
    -------
        list(Atoms) surface supercells with surface formed by first two cell vectors
    """

    supercells = []
    for at_i, at in enumerate(atoms):
        if primitive:
            at = _get_primitive(at, symprec=symprec)

        if simple_cut:
            # surface plane formed by two lattice vectors

            # out of plane vector random or longest better?
            # surf_i = rng.randint(3)
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
            rng = _autopara_per_item_info[at_i]["rng"]

            # surface plane formed by 2 supercell vectors up to max_surface_cell_indices
            # pick 1st vector
            s0 = np.zeros(3, dtype=int)
            while np.sum(np.abs(s0)) == 0:
                s0 = rng.choice(range(-max_surface_cell_indices, max_surface_cell_indices + 1), 3)
            # pick 2nd vector that is not linearly dependent
            s1 = np.zeros(3, dtype=int)
            while np.sum(np.abs(s1)) == 0 or _are_lin_dep(s0, s1):
                s1 = rng.choice(range(-max_surface_cell_indices, max_surface_cell_indices + 1), 3)

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

        if pert != 0.0:
            at.positions += pert * rng.normal(size=at.positions.shape)
        ####################################################################################################
        # workaround for non-working results invalidation when number of atoms has changed
        at.calc = None
        ####################################################################################################

        supercells.append(at)

    return supercells


def surface(*args, **kwargs):
    return autoparallelize(_surface_autopara_wrappable, *args, default_autopara_info={"num_inputs_per_python_subprocess": 10}, **kwargs)
autoparallelize_docstring(surface, _surface_autopara_wrappable, "Atoms")
