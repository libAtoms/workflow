"""
This is testing the ORCA calculator given here, with an example orca output given to test on.

the assets/ workdir contains the orca files and this test depends on them
"""

import os
import shutil
from functools import partial

import pytest

import numpy as np
from ase.build import molecule
from pytest import approx
from pathlib import Path

from ase import Atoms
from ase.calculators.calculator import CalculationFailed 

from wfl.calculators.orca import ORCA, parse_npa_output, natural_population_analysis
from wfl.calculators import generic
from wfl.configset import ConfigSet, OutputSpec
from wfl.autoparallelize import AutoparaInfo

pytestmark = pytest.mark.xfail(reason="calculator needs to be updated to work with latest gitlab version of ase")


ref_parameters = dict(charge=0,
                      mult=1,
                      orca_command="dummy_no_orca_exec",
                      orcablocks='%scf Convergence Tight \nSmearTemp 5000.0 \nmaxiter 500 \n  Rotate \n  {  1,   7,      28.37, 0, 0} \n  {  6,   9,      54.55, 0, 0} \n  {  2,  10,      27.42, 0, 0} \n  {  0,  14,      8.014, 1, 1} \n  {  3,  15,       35.3, 1, 1} \n  {  5,  13,      51.31, 1, 1} \n  {  2,  10,      17.84, 1, 1} \n  {  1,  11,      59.83, 1, 1} \nend \n \nend \n',
                      orcasimpleinput='UHF revPBE def2-TZVP def2/J D3BJ slowconv',
                      task='gradient')

ref_output = dict(energy=-2933.086022884649,
                  forces=np.array([[-1.00418967e+03, -3.42059590e-07, 1.02844648e-05],
                                   [1.00418967e+03, 3.42059590e-07, -1.02844648e-05]]),
                  dipole=np.array([1.44071, -0.0, -0.0])
                  )


def test_orca_calculator():
    orca = ORCA()
    orca.read(os.path.join(os.path.abspath(os.path.dirname(__file__)), '..', 'assets', 'orca'))

    # parameter reading
    for key, val in orca.parameters.items():
        assert ref_parameters[key] == val

    # results reading
    for key, val in orca.results.items():
        # the dipole can bave a little floating point issue
        # because it is written up to 5 digits only
        assert ref_output[key] == approx(val, rel=(1e-4 if key == "dipole" else 1e-7))



def test_orca_is_converged():
    """function to check convergence from orca's output."""

    ref_path = os.path.join(os.path.abspath(os.path.dirname(__file__)), '..', 'assets')
    orca = ORCA()

    orca.label = os.path.join(ref_path, 'orca_scf_converged')
    assert orca.is_converged() is True

    orca.label = os.path.join(ref_path, 'orca_scf_unconverged')
    assert orca.is_converged() is False

    orca.label = os.path.join(ref_path, 'orca_scf_unfinished')
    assert orca.is_converged() is None


@pytest.mark.skipif("ASE_ORCA_COMMAND" not in os.environ, reason="no ORCA executable in path")
def test_full_orca(tmp_path):
    atoms = Atoms("H2", positions=[(0, 0, 0), (0, 0, 0.9)])

    scratchdir = tmp_path / "fake_scratch_dir"
    scratchdir.mkdir(exist_ok=True)
    home_dir = tmp_path / "home_dir"

    # this should raise error and copy over default files
    calc = ORCA(workdir=home_dir, 
                scratchdir=scratchdir, 
                keep_files = False, 
                mult=2)

    atoms.calc = calc
    try:
        atoms.get_potential_energy()
    except CalculationFailed:
        pass
    assert list(scratchdir.iterdir()) == []
    assert home_dir.exists()
    calc_dir = [d for d in home_dir.iterdir()][0]
    for ext in [".ase", ".inp", ".out"]:
        assert (calc_dir / ("orca" + ext)).exists()

    # just check this executes without error
    calc = ORCA(workdir=home_dir, 
                scratchdir=scratchdir, 
                keep_files = "default", 
                mult=1)

    atoms.calc = calc
    atoms.get_potential_energy()


@pytest.mark.skipif("ASE_ORCA_COMMAND" not in os.environ, reason="no ORCA executable in path")
def test_orca_with_generic(tmp_path):
    
    home_dir = tmp_path / "home_dir"

    atoms = Atoms("H2", positions=[(0, 0, 0), (0, 0, 0.9)])
    atoms = [atoms] * 3 + [Atoms("H")]
    inputs = ConfigSet(atoms)
    outputs = OutputSpec()

    calc = ORCA(workdir=home_dir, 
                keep_files = "default", 
                mult=1)

    generic.calculate(inputs=inputs, outputs=outputs, calculator=calc, properties=["energy", "forces"], output_prefix="orca_")


    for at in outputs.to_ConfigSet():
        assert "orca_energy" in at.info or "orca_calculation_failed" in at.info

@pytest.mark.skipif("ASE_ORCA_COMMAND" not in os.environ, reason="no ORCA executable in path")
def test_orca_geometry_optimisation(tmp_path):
    
    home_dir = tmp_path / "home_dir"

    atoms = Atoms("H2", positions=[(0, 0, 0), (0, 0, 0.9)])
    inputs = ConfigSet(atoms)
    outputs = OutputSpec()

    calc = ORCA(workdir=home_dir, 
                keep_files = "default", 
                mult=1, 
                task="opt")


    generic_result = generic.calculate(inputs=inputs, outputs=outputs, calculator=calc, properties=["energy", "forces"], output_prefix="orca_")


    out = list(generic_result)[0]

    assert pytest.approx(out.get_distance(0, 1)) == 0.76812058465248


@pytest.mark.skipif("ASE_ORCA_COMMAND" not in os.environ, reason="no ORCA executable in path")
def test_post_processing(tmp_path):
       
    home_dir = tmp_path / "home_dir"

    atoms = Atoms("H2", positions=[(0, 0, 0), (0, 0, 0.9)])
    calc = ORCA(workdir=home_dir, 
                keep_files = ["*.inp", "*.out", "*.post"], 
                post_process=simplest_orca_post)

    atoms.calc = calc
    atoms.get_potential_energy()
    assert Path(atoms.calc.label + '.post').exists()


def test_parse_npa_output():
    ref_populations = [8.8616525, 0.5691737, 0.5691737]
    ref_charges = [-0.8616525437, 0.4308262720, 0.4308262720]
    fname = os.path.join(os.path.abspath(os.path.dirname(__file__)), '..', 'assets', "orca", "orca.janpa")
    elements, populations, charges = parse_npa_output(fname)
    assert np.all([v1==v2 for v1, v2 in zip(elements, ["O", "H", "H"])])
    assert np.all([pytest.approx(v1)==v2 for v1, v2 in zip(populations, ref_populations)])
    assert np.all([pytest.approx(v1)==v2 for v1, v2 in zip(charges,ref_charges)]) 
    

def simplest_orca_post(orca_calc):

    label = orca_calc.label
    if Path(label + '.out').exists():
        with open(label + ".post", "w") as f:
            f.write("Dummy file generated after ORCA execution\n")


@pytest.mark.skipif("ASE_ORCA_COMMAND" not in os.environ or "JANPA_HOME_DIR" not in os.environ, reason="no ORCA or JANPA executable in path")
def test_run_npa(tmp_path):

    janpa_home_dir = os.environ["JANPA_HOME_DIR"]
    post_func = partial(natural_population_analysis, janpa_home_dir)
       
    home_dir = tmp_path / "home_dir"

    atoms = Atoms("H2", positions=[(0, 0, 0), (0, 0, 0.9)])
    calc = ORCA(workdir=home_dir, 
                keep_files = ["*.inp", "*.out", "*.janpa"], 
                post_process=post_func, 
                )

    inputs = ConfigSet(atoms)
    outputs = OutputSpec()

    generic_results = generic.calculate(inputs=inputs, outputs=outputs, calculator=calc, properties=["energy", "forces"], output_prefix="orca_", raise_calc_exceptions=True)


    atoms = list(generic_results)[0]
    assert "orca_NPA_electron_population" in atoms.arrays
    assert "orca_NPA_charge" in atoms.arrays


ref_freq = {'normal_mode_eigenvalues': np.array([0., 0., 0., 0., 0., 0., -0.60072079, -0.10155918, -0.07241669,
                                                 -0.07238941, 0.14857026, 0.36614641, 0.37056397, 0.41033562,
                                                 0.41034182]),
            'normal_mode_displacements': np.array(
                [[[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
                 [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
                 [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
                 [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
                 [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
                 [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
                 [[-0.0, 0.0, 0.0], [0.0, -0.0, 4.99976e-01], [0.0, 0.0, 5.00019e-01], [0.0, -0.0, -5.01584e-01],
                  [0.0, -0.0, -4.98416e-01]],
                 [[0.0, 0.0, -1.65530e-01], [-0.0, 0.0, 4.93139e-01], [-0.0, 0.0, 4.93060e-01],
                  [-0.0, -0.0, 4.91408e-01], [0.0, -0.0, 4.94797e-01]],
                 [[-5.67750e-02, -8.91300e-02, -0.0], [3.78384e-01, -5.96550e-02, 0.0],
                  [3.75511e-01, -5.71360e-02, 0.0], [-3.87060e-02, 5.90964e-01, 0.0], [-3.86700e-02, 5.87873e-01, 0.0]],
                 [[-8.91340e-02, 5.67820e-02, -0.0], [5.88492e-01, 3.66800e-02, -0.0], [5.90352e-01, 4.06260e-02, -0.0],
                  [-5.84790e-02, -3.77962e-01, 0.0], [-5.82730e-02, -3.75934e-01, 0.0]],
                 [[8.00000e-06, -5.19000e-04, 0.0], [-5.00018e-01, 1.17000e-04, 0.0], [4.99974e-01, 1.72500e-03, 0.0],
                  [7.89000e-04, -4.97828e-01, -0.0], [-8.44000e-04, 5.02166e-01, -0.0]],
                 [[3.94000e-04, -3.90000e-05, 0.0], [-6.71000e-04, -4.99775e-01, -0.0],
                  [-2.27900e-03, 5.00214e-01, -0.0], [-5.00876e-01, -8.01000e-04, 0.0],
                  [4.99126e-01, 8.30000e-04, 0.0]],
                 [[-8.30000e-05, -3.80000e-05, -0.0], [-6.92000e-04, -4.99738e-01, -0.0],
                  [-2.27100e-03, 5.00261e-01, -0.0], [5.01967e-01, 7.96000e-04, 0.0],
                  [-4.98019e-01, -8.67000e-04, 0.0]],
                 [[6.77020e-02, -8.99660e-02, 0.0], [1.94210e-02, 5.61489e-01, 0.0], [1.60870e-02, 5.61639e-01, 0.0],
                  [-4.19742e-01, -2.57180e-02, -0.0], [-4.22479e-01, -2.54060e-02, -0.0]],
                 [[8.99710e-02, 6.77070e-02, 0.0], [2.42810e-02, -4.21873e-01, 0.0], [2.67800e-02, -4.20349e-01, 0.0],
                  [-5.59691e-01, 1.79000e-02, -0.0], [-5.63437e-01, 1.75500e-02, -0.0]]])}


@pytest.mark.skip(reason="Normal mode (eigenvector) reading implemented incorrectly.")
def test_read_frequencies():
    mol = molecule("CH4")
    orca = ORCA(label=os.path.join(os.path.abspath(os.path.dirname(__file__)), '..', 'assets', 'orca_freq_dummy'))
    orca.atoms = mol

    # read the file given
    orca.read_frequencies()

    # results reading
    for key, val in orca.extra_results.items():
        assert ref_freq[key] == approx(val)
