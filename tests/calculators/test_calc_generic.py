import sys
from os.path import join
from io import StringIO
import pytest

import numpy as np
from ase import Atoms
from ase.build import bulk, molecule
from ase.calculators.emt import EMT
from ase.calculators.lj import LennardJones
from pytest import approx

from wfl.calculators import generic
from wfl.configset import ConfigSet, OutputSpec
from wfl.calculators.espresso import Espresso
from wfl.autoparallelize import AutoparaInfo

from tests.calculators.test_qe import espresso_avail

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
    mol_out = generic._run_autopara_wrappable(molecule("CH4"), LennardJones(), properties=["energy", "forces"],
                             output_prefix="lj_dummy_")
    assert isinstance(mol_out, Atoms)
    assert "lj_dummy_energy" in mol_out.info.keys()
    assert mol_out.info["lj_dummy_energy"] == approx(ref_lj_energy)


def test_name_auto():
    # LennardJones is the .__name__ in the calculator
    mol_out = generic._run_autopara_wrappable(molecule("CH4"), LennardJones(), properties=["energy", "forces"], output_prefix="_auto_")
    assert "LennardJones_energy" in mol_out.info.keys()
    assert mol_out.info["LennardJones_energy"] == approx(ref_lj_energy)


def test_atoms_list():
    mol_in = [molecule("CH4"), molecule("CH3")]
    mol_out = generic._run_autopara_wrappable(mol_in, LennardJones(), properties=["energy", "forces"],
                             output_prefix="_auto_")
    assert isinstance(mol_out, list)
    for at in mol_out:
        assert isinstance(at, Atoms)


def test_run(tmp_path):
    mol_in = [molecule("CH4"), molecule("CH4")]
    mol_out = generic.calculate(mol_in, OutputSpec(tmp_path / "run.xyz"), LennardJones(),
                          properties=["energy", "forces"], output_prefix="_auto_")
    assert isinstance(mol_out, ConfigSet)
    for at in mol_out:
        assert isinstance(at, Atoms)


def test_default_properties():
    mol_out = generic._run_autopara_wrappable(bulk("C"), EMT(), output_prefix="dummy_")

    assert "dummy_energy" in mol_out.info.keys()
    assert "dummy_stress" in mol_out.info.keys()
    assert "dummy_forces" in mol_out.arrays.keys()


def test_config_specific_calculator(tmp_path):
    mol_in = [molecule("CH4"), molecule("CH4"), molecule("CH4")]
    mol_in[1].info["WFL_CALCULATOR_KWARGS"] = {'epsilon':2.0}
    mol_in[2].info["WFL_CALCULATOR_INITIALIZER"] = EMT
    calculator = [LennardJones, [], {}]
    mol_out = generic.calculate(mol_in, OutputSpec(tmp_path / "run.xyz"), calculator, properties=["energy", "forces"], output_prefix="dummy_")

    energies = []
    for at in mol_out:
        energies.append(at.info['dummy_energy'])
    assert energies[0] == energies[1]/2 != energies[2]

####################################################################################################

class EMT_override_def_autopara(EMT):
    wfl_generic_default_autopara_info = {"num_inputs_per_python_subprocess": 5}

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

def test_generic_autopara_defaults():
    ats = [Atoms('Al2', positions=[[0,0,0], [1,1,1]], cell=[10]*3, pbc=[True]*3) for _ in range(50)]

    ci = ConfigSet(ats)
    os = OutputSpec()

    # try default
    l_stderr = StringIO()
    sys.stderr = l_stderr
    os = OutputSpec()
    at_proc = generic.calculate(ci, os, EMT())
    sys.stderr = sys.__stderr__
    assert "num_inputs_per_python_subprocess=10" in l_stderr.getvalue()

    # try with class that overrides default
    l_stderr = StringIO()
    sys.stderr = l_stderr
    os = OutputSpec()
    at_proc = generic.calculate(ci, os, EMT_override_def_autopara())
    sys.stderr = sys.__stderr__
    assert "num_inputs_per_python_subprocess=5" in l_stderr.getvalue()

    # again, but with calculator passed as kwargs

    # try default
    l_stderr = StringIO()
    sys.stderr = l_stderr
    os = OutputSpec()
    at_proc = generic.calculate(ci, os, calculator=EMT())
    sys.stderr = sys.__stderr__
    assert "num_inputs_per_python_subprocess=10" in l_stderr.getvalue()

    # try with class that overrides default
    l_stderr = StringIO()
    sys.stderr = l_stderr
    os = OutputSpec()
    at_proc = generic.calculate(ci, os, calculator=EMT_override_def_autopara())
    sys.stderr = sys.__stderr__
    assert "num_inputs_per_python_subprocess=5" in l_stderr.getvalue()

    # try with class that overrides default, and override manually
    l_stderr = StringIO()
    sys.stderr = l_stderr
    os = OutputSpec()
    at_proc = generic.calculate(ci, os, calculator=EMT_override_def_autopara(), autopara_info=AutoparaInfo(num_inputs_per_python_subprocess=3))
    sys.stderr = sys.__stderr__
    assert "num_inputs_per_python_subprocess=3" in l_stderr.getvalue()

@espresso_avail
def test_generic_DFT_autopara_defaults(tmp_path, monkeypatch):
    ats = [Atoms('Al2', positions=[[0,0,0], [1,1,1]], cell=[10]*3, pbc=[True]*3) for _ in range(50)]

    ci = ConfigSet(ats)
    os = OutputSpec()

    l_stderr = StringIO()

    # try with a calculator that overrides an autopara default, namely a DFT calculator
    # that sets num_inputs_per_python_subprocess=1
    sys.stderr = l_stderr
    at_proc = generic.calculate(ci, os, Espresso(workdir=tmp_path))
    sys.stderr = sys.__stderr__
    assert "num_inputs_per_python_subprocess=1" in l_stderr.getvalue()
