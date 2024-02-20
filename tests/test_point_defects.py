import numpy as np
from ase.atoms import Atoms

from wfl.configset import ConfigSet, OutputSpec
from wfl.generate.supercells import vacancy, interstitial, antisite

def test_vacancy_mono():
    at = Atoms('Po', cell=np.eye(3), pbc=[True]*3)

    ci = ConfigSet([at.copy() for _ in range(10)])
    co = OutputSpec()
    vacs = vacancy(ci, co, 64, rng=np.random.default_rng(1))

    assert all([len(at) == 63 for at in vacs])


def test_vacancy_di():
    at = Atoms('Po', cell=np.eye(3), pbc=[True]*3)

    ci = ConfigSet([at.copy() for _ in range(10)])
    co = OutputSpec()
    vacs = vacancy(ci, co, 64, n_vac=2, rng=np.random.default_rng(1))

    assert all([len(at) == 62 for at in vacs])
    assert all([len(at.info['vacancy_Z']) == 2 for at in vacs])


def test_vacancy_cluster_di():
    at = Atoms('Po', cell=2.0 * np.eye(3), pbc=[True]*3)

    ci = ConfigSet([at.copy() for _ in range(10)])
    co = OutputSpec()
    vacs = vacancy(ci, co, 64, rng=np.random.default_rng(1), n_vac=2, cluster_r=1.5)

    for at in vacs:
        assert len(at) == 62
        assert len(at.info['vacancy_Z'] == 2)

        dp = at.info['vacancy_pos'][0] - at.info['vacancy_pos'][1]
        dp_s = dp @ at.cell.reciprocal().T
        dp_s -= np.round(dp_s)
        dp = dp_s @ at.cell
        assert np.linalg.norm(dp) <= 2.0 * 1.5


def test_vacancy_not_enough():
    at = Atoms('Po', cell=2.0 * np.eye(3), pbc=[True]*3)
    at *= 2

    ci = ConfigSet([at.copy() for _ in range(10)])
    co = OutputSpec()
    vacs = vacancy(ci, co, 8, rng=np.random.default_rng(1), n_vac=8, cluster_r=1.5)

    # not enough total atoms
    for at in vacs:
        assert len(at) == 8
        assert 'vacancy_Z' not in at.info

    ci = ConfigSet([at.copy() for _ in range(10)])
    co = OutputSpec()
    vacs = vacancy(ci, co, 64, rng=np.random.default_rng(1), n_vac=8, cluster_r=1.1)

    # not enough total within cutoff
    for at in vacs:
        assert len(at) == 64
        assert 'vacancy_Z' not in at.info


def test_interstitial():
    at = Atoms('Po', cell=np.eye(3), pbc=[True]*3)

    ci = ConfigSet([at.copy() for _ in range(10)])
    co = OutputSpec()
    intersts = interstitial(ci, co, 64, rng=np.random.default_rng(1))

    assert all([len(at) == 65 for at in intersts])


def test_antisite():
    at = Atoms('PoCd', positions=[[0, 0, 0], [0.5, 0.5, 0.5]], cell=np.eye(3), pbc=[True]*3)

    ci = ConfigSet([at.copy() for _ in range(10)])
    co = OutputSpec()
    intersts = antisite(ci, co, 64, rng=np.random.default_rng(1))

    assert all([len(at) == 54 for at in intersts])
