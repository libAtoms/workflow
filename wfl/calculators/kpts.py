"""Universal k-spacing functions
"""

import numpy as np


def universal_kspacing_n_k(cell, pbc, kspacing):
    """Calculate universal kspacing, but never > 1 for pbc False

    Parameters
    ----------
    kspacing: float
        spacing in reciprocal space (assuming magnitude of reciprical lattice vectors includes factor of 2 pi
    cell: ase.cell.Cell
        cell of atomic configuration
    pbc: (bool, bool, bool)
        Whether direction is periodic and should be k-point sampled

    Returns
    -------
    kmesh: list((float, float, float)) positions of BZ points in reciprocal lattice coordinates
    """
    n_k = np.ceil(np.linalg.norm(2.0 * np.pi * cell.reciprocal(), axis=1) / kspacing).astype(int)
    n_k[np.logical_not(pbc)] = 1

    return n_k


def universal_kspacing_k_mesh(cell, pbc, kspacing, kgamma=True, inversion_sym=False):
    """Calculate universal kspacing, but never > 1 for pbc False

    Parameters
    ----------
    kspacing: float
        spacing in reciprocal space (assuming magnitude of reciprical lattice vectors includes factor of 2 pi
    cell: ase.cell.Cell
        cell of atomic configuration
    pbc: (bool, bool, bool)
        Whether direction is periodic and should be k-point sampled
    kgamma: bool, default True
        use Gamma centered mesh
    inversion_sym: bool, default False
        use inversion symmetry to fold k-points

    Returns
    -------
    kmesh: list((float, float, float)) positions of BZ points in reciprocal lattice coordinates
    """
    # print("recip lattice mags", np.linalg.norm(2.0 * np.pi * cell.reciprocal(), axis=1))
    n_k = universal_kspacing_n_k(cell, pbc, kspacing)

    k_mesh_int_max = n_k // 2
    k_mesh_int_min = - k_mesh_int_max
    k_mesh_int_min += 1 - (n_k % 2)

    k_mesh_int = [np.arange(k_mesh_int_min[ii], k_mesh_int_max[ii] + 1) for ii in range(0, 3)]

    # print("k_mesh_int")
    # print(k_mesh_int[0])
    # print(k_mesh_int[1])
    # print(k_mesh_int[2])

    k_mesh = k_mesh_int

    if not kgamma:
        k_shift_inds = np.where(n_k % 2 == 0)[0]
        k_mesh = [k_mesh[ii] - (0.5 if ii in k_shift_inds else 0) for ii in range(0, 3)]

    k_mesh = [k_mesh[ii] / n_k[ii] for ii in range(0, 3)]

    # print("k_mesh final")
    # print(n_k[0], k_mesh[0])
    # print(n_k[1], k_mesh[1])
    # print(n_k[2], k_mesh[2])

    k_mesh = np.meshgrid(*k_mesh)
    k_mesh = np.asarray([k_mesh[ii].flatten() for ii in range(0, 3)]).T

    k_weights = np.ones(k_mesh.shape[0])
    if inversion_sym:
        del_ind = []
        dup_is = [-1] + [np.argmin(np.linalg.norm(k_mesh[:ii] + k_mesh[ii], axis=1))
                               if np.min(np.linalg.norm(k_mesh[:ii] + k_mesh[ii], axis=1)) < 1.0e-6 else -1
                               for ii in range(1, len(k_mesh))]
        dup_is = np.asarray(dup_is)
        k_weights[dup_is[np.where(dup_is >= 0)]] += 1
        # print("dup_is", dup_is)
        # print("k_weights", k_weights)
        k_mesh = k_mesh[np.where(dup_is < 0)]
        k_weights = k_weights[np.where(dup_is < 0)]
        k_mesh = np.concatenate((k_mesh, k_weights.reshape(-1, 1)), axis=1)

    return k_mesh
