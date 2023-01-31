import os
import numpy as np
import pytest
import shutil

from ase.io import write
from ase.build import molecule
from ase.calculators.mopac import MOPAC

from wfl.calculators import generic
from wfl.calculators.mopac import MOPAC
from wfl.configset import ConfigSet, OutputSpec

ref_energy = -0.38114618938803013
ref_forces = np.array([[ 0.00598923, -0.00306901, -0.01411094],
                        [ 0.00374263, -0.00137642,  0.01242343],
                        [ 0.00710343, -0.00450089,  0.00556635],
                        [-0.00226356, -0.00497725,  0.00713461],
                        [-0.01457173,  0.01392352, -0.0110134 ]])
ref_pos = np.array([[-0.14171658, -0.08539037, -0.06833327],
                    [ 0.68255623,  0.39319224,  0.51566672],
                    [-0.64855418, -0.61104146,  0.50794689],
                    [ 0.50823426, -0.74095259, -0.65120197],
                    [-0.7545869,   0.4412089,  -0.47989706]])

@pytest.fixture
def atoms():
    atoms = molecule("CH4")
    atoms.rattle(stdev=0.1, seed=1305)
    return atoms

@pytest.mark.skipif(not shutil.which("mopac") and "ASE_MOPAC_COMMAND" not in os.environ, 
                reason="mopac not in PATH and command not given")
def test_ase_mopac(tmp_path, atoms):
    """test that the regular MOPAC works, since there was a recent bug fix
        Should be returning final heat of formation for "energy", e.g.: 
                    self.results['energy'] = self.results['final_hof']
        This step was missing at some point which failed with 
        PropertyNotImplementedError"""

    os.chdir(tmp_path)

    atoms.calc = MOPAC(label='tmp', method="AM1", task="GRADIENTS")

    assert np.allclose(atoms.get_potential_energy(), ref_energy)
    assert np.allclose(atoms.get_forces(), ref_forces)

@pytest.mark.skipif(not shutil.which("mopac") and "ASE_MOPAC_COMMAND" not in os.environ, 
                reason="mopac not in PATH and command not given")
def test_wfl_mopac(tmp_path, atoms):
    fn_in = tmp_path / 'mopac_in.xyz' 

    write(fn_in, atoms)

    calc = (MOPAC, [], {"workdir":tmp_path, "method": "AM1", "task":"GRADIENTS"})

    configs_eval = generic.run(
        inputs=ConfigSet(fn_in),
        outputs = OutputSpec(),
        calculator = calc, 
        output_prefix="mopac_"
    )

    at = list(configs_eval)[0]
    assert np.allclose(at.info["mopac_energy"], ref_energy)
    assert np.allclose(at.arrays["mopac_forces"], ref_forces, rtol=1e-4)

