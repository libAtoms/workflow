import sys
import numpy as np
import pytest

from ase.atoms import Atoms
import ase.io

from wfl.configset import ConfigSet, OutputSpec
from wfl.generate.phonopy import phonopy
try:
    import phono3py
except ImportError:
    phono3py = None


def test_phonopy(tmp_path):
    at = Atoms(numbers=[29], cell = [[0, 2, 2], [2, 0, 2], [2, 2, 0]], positions = [[0, 0, 0]], pbc = [True]*3)

    ci = ConfigSet([at])
    co = OutputSpec(tmp_path / "phonopy_2.xyz")

    sc = at * [3, 3, 3]
    displs = [0.1, 0.2]
    strain_displs = [0.05, 0.1]
    pert = phonopy(ci, co, displs, strain_displs, [3, 3, 3])

    pert = list(pert)

    # displaced + 1 phonon x 2 + 6 strain x 2
    assert len(pert) == 1 + 1*2 + 6*2

    # first 3 configs should be undisplaced and 2 displaced magnitudes along (1 1 0)
    for d, at in zip([0.0] + displs, pert[:3]):
        # first atom displaced along (1 1 0)
        np.testing.assert_allclose(at.positions[0], [0.0, d/np.sqrt(2), d/np.sqrt(2)])
        # all other atoms undisplaced
        for v in at.positions[1:]:
            assert min(np.linalg.norm(sc.positions[1:] - v, axis=1)) < 1.0e-7

@pytest.mark.skipif(phono3py is None, reason="No phono3py module")
def test_phono3py(tmp_path):
    at0 = Atoms(numbers=[29], cell = [[0, 2, 2], [2, 0, 2], [2, 2, 0]], positions = [[0, 0, 0]], pbc = [True]*3)
    at1 = Atoms(numbers=[29], cell = [[0, 1.9, 1.9], [1.9, 0, 1.9], [1.9, 1.9, 0]], positions = [[0, 0, 0]], pbc = [True]*3)

    ci = ConfigSet([at0, at1])
    co = OutputSpec(tmp_path / "phonopy_3.xyz")

    displs = [0.1, 0.2]
    strain_displs = [0.05, 0.1]
    pert = phonopy(ci, co, displs, strain_displs, [3, 3, 3], [2, 2, 2], 3.0)

    pert = list(pert)

    # (undisplaced fc2 and fc3 + (1 harmonic + 13 cubic + 6 strain) * 2 magnitudes) * 2 configs
    assert len(pert) == 4 + ((1 + 13 + 6) * 2 ) * 2
    assert sum([at.info["config_type"] == "phonon_harmonic_0" for at in pert]) == 1*2
    assert sum([at.info["config_type"] == "phonon_harmonic_1" for at in pert]) == 1*2
    assert sum([at.info["config_type"] == "phonon_strain_0" for at in pert]) == 6*2
    assert sum([at.info["config_type"] == "phonon_strain_1" for at in pert]) == 6*2

    for at in pert:
        if at.info["config_type"].startswith("phonon_harmonic_") and "undispl" not in at.info["config_type"]:
            di = int(at.info["config_type"].replace("phonon_harmonic_", ""))
            np.testing.assert_allclose(at.positions[0], [0.0, displs[di] / np.sqrt(2), displs[di] / np.sqrt(2)])

    assert sum([at.info["config_type"] == "phonon_cubic_0" for at in pert]) == 13*2
    assert sum([at.info["config_type"] == "phonon_cubic_1" for at in pert]) == 13*2


@pytest.mark.skipif(phono3py is None, reason="No phono3py module")
def test_phono3py_same_supercell(tmp_path):
    at0 = Atoms(numbers=[29], cell = [[0, 2, 2], [2, 0, 2], [2, 2, 0]], positions = [[0, 0, 0]], pbc = [True]*3)
    at1 = Atoms(numbers=[29], cell = [[0, 1.9, 1.9], [1.9, 0, 1.9], [1.9, 1.9, 0]], positions = [[0, 0, 0]], pbc = [True]*3)

    ci = ConfigSet([at0, at1])
    co = OutputSpec(tmp_path / "phonopy_3.xyz")

    displs = [0.1, 0.2]
    strain_displs = [0.05, 0.1]
    pert = phonopy(ci, co, displs, strain_displs, [2, 2, 2], [2, 2, 2], 3.0)

    pert = list(pert)

    # (undisplaced fc2 + (1 harmonic + 13 cubic + 6 strain) * 2 magnitudes) * 2 configs
    assert len(pert) == 2 + ((1 + 13 + 6) * 2 ) * 2
