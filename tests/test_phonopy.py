import sys
import numpy as np

from ase.atoms import Atoms
import ase.io

from wfl.configset import ConfigSet_in, ConfigSet_out
from wfl.generate_configs.phonopy import run


def test_phonopy(tmp_path):
    at = Atoms(numbers=[29], cell = [[0, 2, 2], [2, 0, 2], [2, 2, 0]], positions = [[0, 0, 0]], pbc = [True]*3)

    ci = ConfigSet_in(input_configs=[at])
    co = ConfigSet_out(output_files=str(tmp_path / "phonopy_2.xyz"))

    sc = at * [3, 3, 3]
    displs = [0.1, 0.2]
    strain_displs = [0.05, 0.1]
    pert = run(ci, co, displs, strain_displs, [3, 3, 3])

    pert = list(pert)

    assert len(pert) == 1*2 + 6*2

    for d, at in zip(displs, pert):
        np.testing.assert_allclose(at.positions[0], [0.0, d/np.sqrt(2), d/np.sqrt(2)])
        for v in at.positions[1:]:
            assert min(np.linalg.norm(sc.positions[1:] - v, axis=1)) < 1.0e-7


def test_phono3py(tmp_path):
    at0 = Atoms(numbers=[29], cell = [[0, 2, 2], [2, 0, 2], [2, 2, 0]], positions = [[0, 0, 0]], pbc = [True]*3)
    at1 = Atoms(numbers=[29], cell = [[0, 1.9, 1.9], [1.9, 0, 1.9], [1.9, 1.9, 0]], positions = [[0, 0, 0]], pbc = [True]*3)

    ci = ConfigSet_in(input_configs=[at0, at1])
    co = ConfigSet_out(output_files=str(tmp_path / "phonopy_3.xyz"))

    displs = [0.1, 0.2]
    strain_displs = [0.05, 0.1]
    pert = run(ci, co, displs, strain_displs, [3, 3, 3], [2, 2, 2], 3.0)

    pert = list(pert)

    assert len(pert) == ((1 + 13 + 6) * 2 ) * 2
    assert sum([at.info["config_type"] == "phonon_harmonic_0" for at in pert]) == 1*2
    assert sum([at.info["config_type"] == "phonon_harmonic_1" for at in pert]) == 1*2
    assert sum([at.info["config_type"] == "phonon_strain_0" for at in pert]) == 6*2
    assert sum([at.info["config_type"] == "phonon_strain_1" for at in pert]) == 6*2

    for at in pert:
        if at.info["config_type"].startswith("phonon_harmonic_"):
            di = int(at.info["config_type"].replace("phonon_harmonic_", ""))
            np.testing.assert_allclose(at.positions[0], [0.0, displs[di] / np.sqrt(2), displs[di] / np.sqrt(2)])

    assert sum([at.info["config_type"] == "phonon_cubic_0" for at in pert]) == 13*2
    assert sum([at.info["config_type"] == "phonon_cubic_1" for at in pert]) == 13*2
