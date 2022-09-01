from os.path import join

import numpy as np
from ase import Atoms
from ase.build import bulk, molecule
from ase.calculators.emt import EMT
from ase.calculators.lj import LennardJones
from pytest import approx

from wfl.calculators import generic
from wfl.configset import ConfigSet, OutputSpec

ref_lj_energy = -4.52573996914352
ref_morse_energy = -3.4187397762024867

ref_lj_forces = np.array([[0., 0., 0.],
                          [0.91628394, 0.91628394, 0.91628394],
                          [-0.91628394, -0.91628394, 0.91628394],
                          [0.91628394, -0.91628394, -0.91628394],
                          [-0.91628394, 0.91628394, -0.91628394]])
ref_morse_forces = np.array([[0., 0., 0.],
                             [-1.83980777, -1.83980777, -1.83980777],
                             [1.83980777, 1.83980777, -1.83980777],
                             [-1.83980777, 1.83980777, 1.83980777],
                             [1.83980777, -1.83980777, 1.83980777]])


def test_calculator_generic():
    mol_out = generic.run_autopara_wrappable(molecule("CH4"), LennardJones(), properties=["energy", "forces"],
                             output_prefix="lj_dummy_")
    assert isinstance(mol_out, Atoms)
    assert "lj_dummy_energy" in mol_out.info.keys()
    assert mol_out.info["lj_dummy_energy"] == approx(ref_lj_energy)


def test_name_auto():
    # LennardJones is the .__name__ in the calculator
    mol_out = generic.run_autopara_wrappable(molecule("CH4"), LennardJones(), properties=["energy", "forces"], output_prefix="_auto_")
    assert "LennardJones_energy" in mol_out.info.keys()
    assert mol_out.info["LennardJones_energy"] == approx(ref_lj_energy)


def test_atoms_list():
    mol_in = [molecule("CH4"), molecule("CH3")]
    mol_out = generic.run_autopara_wrappable(mol_in, LennardJones(), properties=["energy", "forces"],
                             output_prefix="_auto_")
    assert isinstance(mol_out, list)
    for at in mol_out:
        assert isinstance(at, Atoms)


def test_run(tmp_path):
    mol_in = [molecule("CH4"), molecule("CH4")]
    mol_out = generic.run(mol_in, OutputSpec(tmp_path / "run.xyz"), LennardJones(),
                          properties=["energy", "forces"], output_prefix="_auto_")
    assert isinstance(mol_out, ConfigSet)
    for at in mol_out:
        assert isinstance(at, Atoms)


def test_default_properties():
    mol_out = generic.run_autopara_wrappable(bulk("C"), EMT(), output_prefix="dummy_")

    assert "dummy_energy" in mol_out.info.keys()
    assert "dummy_stress" in mol_out.info.keys()
    assert "dummy_forces" in mol_out.arrays.keys()
