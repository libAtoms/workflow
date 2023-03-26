import os
import numpy as np
import pytest
import shutil

from ase.io import write
from ase.build import molecule
from ase.calculators.mopac import MOPAC as ASE_MOPAC

from wfl.calculators import generic
from wfl.calculators.mopac import MOPAC as WFL_MOPAC
from wfl.configset import ConfigSet, OutputSpec

ref_opt_energy = -0.38114618938803013

ref_relaxed_positions = np.array([[-0.09407857, -0.10925283, -0.04563854],
                                [ 0.63451195,  0.48912991,  0.5428456 ],
                                [-0.70392575, -0.73498716,  0.64129948],
                                [ 0.45409285, -0.76405965, -0.75748043],
                                [-0.7614799 ,  0.57469914, -0.61249409]])


ref_opt_forces = np.array([[ 0.0059888,  -0.00308111, -0.01411684],
                        [ 0.00374831, -0.00137156,  0.01242633],
                        [ 0.00710269, -0.00450137,  0.00556648],
                        [-0.00226703, -0.00497165,  0.00713934],
                        [-0.01457277,  0.01392569, -0.01101531]])


ref_single_point_energy = 1.5967898933159592
ref_single_point_forces = np.array([[11.12550451, -1.37163648,  0.56233512],
                                    [-0.02648502,  1.0405767,  0.38785587],
                                    [-4.96900984, -5.79515005,  6.32239103],
                                    [ 0.29056343, -0.80075317, -1.56623281],
                                    [-6.42057309,  6.92696301, -5.70634926]])

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

    atoms.calc = ASE_MOPAC(label='tmp', method="AM1", task="1SCF GRADIENTS")

    assert np.allclose(atoms.get_potential_energy(), ref_single_point_energy)
    assert np.allclose(atoms.get_forces(), ref_single_point_forces)

@pytest.mark.skipif(not shutil.which("mopac") and "ASE_MOPAC_COMMAND" not in os.environ, 
                reason="mopac not in PATH and command not given")
def test_wfl_mopac_geo_opt_calc(tmp_path, atoms):

    os.chdir(tmp_path)

    atoms.calc = WFL_MOPAC(label='tmp', task="AM1 EF GRADIENTS RELSCF=0.001")

    assert np.allclose(atoms.get_potential_energy(), ref_opt_energy)
    assert pytest.approx(atoms.get_forces(), abs=1e-8) == ref_opt_forces
    assert np.allclose(atoms.calc.extra_results["relaxed_positions"], ref_relaxed_positions)

@pytest.mark.skipif(not shutil.which("mopac") and "ASE_MOPAC_COMMAND" not in os.environ, 
                reason="mopac not in PATH and command not given")
def test_wfl_mopac_geo_opt_generic(tmp_path, atoms):


    fn_in = tmp_path / 'mopac_in.xyz' 

    write(fn_in, atoms)

    calc = (WFL_MOPAC, [], {"workdir":tmp_path, "task":"AM1 EF GRADIENTS RELSCF=0.001"})

    configs_eval = generic.run(
        inputs=ConfigSet(fn_in),
        outputs = OutputSpec(),
        calculator = calc, 
        output_prefix="mopac_"
    )

    at = list(configs_eval)[0]
    assert np.allclose(at.info["mopac_energy"], ref_opt_energy)
    assert pytest.approx(at.arrays["mopac_forces"], abs=1e-7) == ref_opt_forces
    assert np.allclose(at.positions, ref_relaxed_positions)


@pytest.mark.skipif(not shutil.which("mopac") and "ASE_MOPAC_COMMAND" not in os.environ, 
                reason="mopac not in PATH and command not given")
def test_wfl_mopac_calc(tmp_path, atoms):

    os.chdir(tmp_path)

    atoms.calc = WFL_MOPAC(label='tmp', task="AM1 1SCF GRADIENTS RELSCF=0.001")

    assert np.allclose(atoms.get_potential_energy(), ref_single_point_energy)
    assert pytest.approx(atoms.get_forces(), abs=1e-8) == ref_single_point_forces 


@pytest.mark.skipif(not shutil.which("mopac") and "ASE_MOPAC_COMMAND" not in os.environ, 
                reason="mopac not in PATH and command not given")
def test_wfl_mopac_generic(tmp_path, atoms):


    fn_in = tmp_path / 'mopac_in.xyz' 

    write(fn_in, atoms)

    calc = (WFL_MOPAC, [], {"workdir":tmp_path, "task":"AM1 1SCF GRADIENTS RELSCF=0.001"})

    configs_eval = generic.run(
        inputs=ConfigSet(fn_in),
        outputs = OutputSpec(),
        calculator = calc, 
        output_prefix="mopac_"
    )

    at = list(configs_eval)[0]
    assert np.allclose(at.info["mopac_energy"], ref_single_point_energy)
    assert pytest.approx(at.arrays["mopac_forces"], abs=1e-6) == ref_single_point_forces 
