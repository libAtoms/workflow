import numpy as np

from ase.cell import Cell
from wfl.calculators.kpts import universal_kspacing_n_k, universal_kspacing_k_mesh

def test_kspacing_n_k():
    cell = Cell([[5.0, 0, 0], [0, 2.5, 0], [0, 0, 2.5]])
    assert np.all(universal_kspacing_n_k(cell, [True] * 3, 0.7) == [2, 4, 4])

    cell = Cell([[5.0, 0, 0], [0, 3.5, 0], [0, 0, 2.5]])
    assert np.all(universal_kspacing_n_k(cell, [True] * 3, 0.7) == [2, 3, 4])

    cell = Cell([[10.0, 0, 0], [0, 3.5, 0], [0, 0, 2.5]])
    assert np.all(universal_kspacing_n_k(cell, [True] * 3, 0.7) == [1, 3, 4])


def test_kspacing_k_mesh():
    cell = Cell([[5.0, 0, 0], [0, 2.5, 0], [0, 0, 2.5]])

    k_mesh = universal_kspacing_k_mesh(cell, [True] * 3, 0.7, kgamma=True)
    assert k_mesh.shape == (32, 3)
    assert np.min(np.linalg.norm(k_mesh[:, 0:3], axis=1)) == 0.0

    k_mesh = universal_kspacing_k_mesh(cell, [True] * 3, 0.7, kgamma=True, inversion_sym=True)
    assert k_mesh.shape == (28, 4)
    assert np.min(np.linalg.norm(k_mesh[:, 0:3], axis=1)) == 0.0
    assert np.sum(k_mesh[:, 3]) == 32

    k_mesh = universal_kspacing_k_mesh(cell, [True] * 3, 0.7, kgamma=False)
    assert k_mesh.shape == (32, 3)
    assert np.min(np.linalg.norm(k_mesh[:, 0:3], axis=1)) > 0.0

    k_mesh = universal_kspacing_k_mesh(cell, [True] * 3, 0.7, kgamma=False, inversion_sym=True)
    k_mesh.shape == (16, 4)
    assert np.min(np.linalg.norm(k_mesh[:, 0:3], axis=1)) > 0.0
    assert np.sum(k_mesh[:, 3]) == 32

    cell=Cell([[5.0, 0, 0], [0, 3.5, 0], [0, 0, 2.5]])

    k_mesh = universal_kspacing_k_mesh(cell, [True] * 3, 0.7, kgamma=True)
    assert k_mesh.shape == (24, 3)
    assert np.min(np.linalg.norm(k_mesh[:, 0:3], axis=1)) == 0.0

    k_mesh = universal_kspacing_k_mesh(cell, [True] * 3, 0.7, kgamma=True, inversion_sym=True)
    assert k_mesh.shape == (20, 4)
    assert np.min(np.linalg.norm(k_mesh[:, 0:3], axis=1)) == 0.0
    assert np.sum(k_mesh[:, 3]) == 24

    k_mesh = universal_kspacing_k_mesh(cell, [True] * 3, 0.7, kgamma=False)
    assert k_mesh.shape == (24, 3)
    assert np.min(np.linalg.norm(k_mesh[:, 0:3], axis=1)) > 0.0

    k_mesh = universal_kspacing_k_mesh(cell, [True] * 3, 0.7, kgamma=False, inversion_sym=True)
    k_mesh.shape == (12, 4)
    assert np.min(np.linalg.norm(k_mesh[:, 0:3], axis=1)) > 0.0
    assert np.sum(k_mesh[:, 3]) == 24

    cell=Cell([[10.0, 0, 0], [0, 3.5, 0], [0, 0, 2.5]])

    k_mesh = universal_kspacing_k_mesh(cell, [True] * 3, 0.7, kgamma=True)
    assert k_mesh.shape == (12, 3)
    assert np.all(k_mesh[:, 0] == 0)

    k_mesh = universal_kspacing_k_mesh(cell, [True] * 3, 0.7, kgamma=True, inversion_sym=True)
    assert k_mesh.shape == (8, 4)
    assert np.all(k_mesh[:, 0] == 0)
    assert np.sum(k_mesh[:, 3]) == 12

    k_mesh = universal_kspacing_k_mesh(cell, [True] * 3, 0.7, kgamma=False)
    assert k_mesh.shape == (12, 3)
    assert np.all(k_mesh[:, 0] == 0)

    k_mesh = universal_kspacing_k_mesh(cell, [True] * 3, 0.7, kgamma=False, inversion_sym=True)
    assert k_mesh.shape == (6, 4)
    assert np.all(k_mesh[:, 0] == 0)
    assert np.sum(k_mesh[:, 3]) == 12


def test_kspacing_k_mesh_non_periodic():
    cell = Cell([[5.0, 0, 0], [0, 2.5, 0], [0, 0, 2.5]])

    assert np.all(universal_kspacing_n_k(cell, [True, True, False], 0.7) == [2, 4, 1])

    k_mesh = universal_kspacing_k_mesh(cell, [True, True, False], 0.7, kgamma=True)
    assert k_mesh.shape == (8, 3)
    assert np.min(np.linalg.norm(k_mesh[:, 0:3], axis=1)) == 0.0

    k_mesh = universal_kspacing_k_mesh(cell, [True, True, False], 0.7, kgamma=True, inversion_sym=True)
    assert k_mesh.shape == (7, 4)
    assert np.min(np.linalg.norm(k_mesh[:, 0:3], axis=1)) == 0.0
    assert np.sum(k_mesh[:, 3]) == 8

    k_mesh = universal_kspacing_k_mesh(cell, [True, True, False], 0.7, kgamma=False)
    assert k_mesh.shape == (8, 3)
    assert np.min(np.linalg.norm(k_mesh[:, 0:3], axis=1)) > 0.0

    k_mesh = universal_kspacing_k_mesh(cell, [True, True, False], 0.7, kgamma=False, inversion_sym=True)
    k_mesh.shape == (4, 4)
    assert np.min(np.linalg.norm(k_mesh[:, 0:3], axis=1)) > 0.0
    assert np.sum(k_mesh[:, 3]) == 8

