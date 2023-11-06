"""
Tests for Quantum Espresso calculator interface
"""
import os
from shutil import which, copy as shutil_copy
from pathlib import Path
import pytest

import ase.io
import numpy as np
import requests
from ase import Atoms
from ase.build import bulk
from pytest import approx, fixture, raises, skip

# from wfl.calculators.espresso import evaluate_autopara_wrappable, qe_kpoints_and_kwargs
import wfl.calculators.espresso
from wfl.calculators import generic
from wfl.configset import ConfigSet, OutputSpec
from wfl.autoparallelize import AutoparaInfo


@fixture(scope="session")
def qe_cmd_and_pseudo(tmp_path_factory):
    """Quantum Espresso fixture

    - checks if pw.x exists (skip otherwise)
    - downloads a pseudo-potential for Si

    implementation based on:
    https://stackoverflow.com/questions/63417661/pytest-downloading-a-test-file-once-and-using-it-for-multiple-tests

    Returns
    -------
    cmd: str
        command for pw.x
    pspot_file: str
        Si pseudo potential file
    """

    if not which("pw.x"):
        skip("no pw.x executable")
    else:
        cmd = which("pw.x")

    # originally downloaded from here, but broken due to need for account/license click
    # url = "https://www.quantum-espresso.org/upf_files/Si.pbe-n-kjpaw_psl.1.0.0.UPF"
    # replaced with this
    # url = "http://nninc.cnf.cornell.edu/psp_files/Si.pz-vbc.UPF"
    # alternative
    # url = "https://web.mit.edu/espresso_v6.1/amd64_linux26/qe-6.1/pseudo/Si.pz-vbc.UPF"

    # current version is using a local copy, from the web.mit.edu URL above

    # write to a temporary file
    pspot_file = tmp_path_factory.getbasetemp() / "Si.UPF"
    shutil_copy(Path(__file__).parent / ".." / "assets" / "QE" / "Si.pz-vbc.UPF", pspot_file)

    return cmd, pspot_file


def test_qe_kpoints(tmp_path, qe_cmd_and_pseudo):

    qe_cmd, pspot = qe_cmd_and_pseudo


    kw = dict(
        pseudopotentials=dict(Si=os.path.basename(pspot)),
        input_data={"SYSTEM": {"ecutwfc": 40, "input_dft": "LDA",}},
        pseudo_dir=os.path.dirname(pspot),
        kpts=(2, 3, 4),
        conv_thr=0.0001,
        workdir=tmp_path
    ) 

    # PBC = TTT
    atoms = Atoms("H", cell=[1, 1, 1], pbc=True)
    properties = ["energy", "stress"] 
    calc = wfl.calculators.espresso.Espresso(**kw)
    calc.atoms = atoms.copy()
    calc.setup_calc_params(properties)

    assert "tstress" in calc.parameters
    assert calc.parameters['kpts'] == (2, 3, 4)

    # PBC = FFF
    atoms = Atoms("H", cell=[1, 1, 1], pbc=False)
    properties = ["energy", "stress", "forces"] 
    calc = wfl.calculators.espresso.Espresso(**kw)
    calc.atoms = atoms.copy()
    properties = calc.setup_calc_params(properties)

    assert "tstress" in calc.parameters
    assert not calc.parameters["tstress"]

    assert "tprnfor" in calc.parameters
    assert calc.parameters["tprnfor"]

    assert "stress" not in properties
    assert calc.parameters["kpts"] is None
    assert calc.parameters["kspacing"] is None
    assert calc.parameters["koffset"] is False

    # PBC mixed -- kpts
    atoms = Atoms("H", cell=[1, 1, 1], pbc=[True, False, False])
    properties = ["energy", "stress", "forces"] 
    kw["koffset"] = True
    calc = wfl.calculators.espresso.Espresso(**kw)
    calc.atoms = atoms.copy()
    properties = calc.setup_calc_params(properties)

    assert "tstress" in calc.parameters
    assert not calc.parameters["tstress"]

    assert "tprnfor" in calc.parameters
    assert calc.parameters["tprnfor"]

    assert "stress" not in properties

    assert calc.parameters["kpts"] == (2, 1, 1)
    assert calc.parameters["kspacing"] is None
    assert calc.parameters["koffset"] == (1, 0, 0)


    # koffset in mixed PBC
    atoms = Atoms("H", cell=[1, 1, 1], pbc=[True, False, False])
    properties = ["energy", "forces"] 
    kw["koffset"] = False 
    calc = wfl.calculators.espresso.Espresso(**kw)
    calc.atoms = atoms.copy()
    properties = calc.setup_calc_params(properties)

    assert calc.parameters["koffset"] is False


    # PBC mixed -- kspacing
    atoms = Atoms("H", cell=[1, 1, 1], pbc=[True, False, False])
    properties = ["energy", "stress", "forces"] 
    kw["kspacing"] = 0.1
    kw["koffset"] = (0, 1, 0)
    calc = wfl.calculators.espresso.Espresso(**kw)
    calc.atoms = atoms.copy()
    properties = calc.setup_calc_params(properties)

    assert "tstress" in calc.parameters
    assert not calc.parameters["tstress"]

    assert "tprnfor" in calc.parameters
    assert calc.parameters["tprnfor"]

    assert "stress" not in properties
    assert calc.parameters["kpts"] == (63, 1, 1)
    assert calc.parameters["kspacing"] is None
    assert calc.parameters["koffset"] == (0, 0, 0)


def test_qe_calculation(tmp_path, qe_cmd_and_pseudo):
    # command and pspot
    qe_cmd, pspot = qe_cmd_and_pseudo

    # atoms
    at = bulk("Si")
    at.positions[0, 0] += 0.01
    at0 = Atoms("Si", cell=[6.0, 6.0, 6.0], positions=[[3.0, 3.0, 3.0]], pbc=False)

    kw = dict(
        pseudopotentials=dict(Si=os.path.basename(pspot)),
        input_data={"SYSTEM": {"ecutwfc": 40, "input_dft": "LDA",}},
        pseudo_dir=os.path.dirname(pspot),
        kpts=(2, 2, 2),
        conv_thr=0.0001,
        calculator_command=qe_cmd,
        workdir=tmp_path
    ) 

    calc = (wfl.calculators.espresso.Espresso, [], kw)

    # output container
    c_out = OutputSpec("qe_results.xyz", file_root=tmp_path)

    results = generic.calculate(
        inputs=[at0, at],
        outputs=c_out, 
        calculator=calc, 
        output_prefix='QE_',
    )


    # unpack the configset
    si_single, si2 = list(results)

    # dev: type hints
    si_single: Atoms
    si2: Atoms

    # single atoms tests
    assert "QE_stress" not in si_single.info
    assert "QE_energy" in si_single.info
    assert si_single.info["QE_energy"] == approx(expected=-101.20487969465684, abs=1e-2)
    assert si_single.get_volume() == approx(6.0 ** 3)

    # bulk Si tests
    assert "QE_energy" in si2.info
    assert si2.info["QE_energy"] == approx(expected=-213.10730256386654, abs=1e-3)
    assert "QE_stress" in si2.info
    print(si2.info["QE_stress"])
    assert si2.info["QE_stress"] == approx(
        abs=1e-3,
        expected = np.array([-0.03510667, -0.03507546, -0.03507546, -0.00256625, -0., -0.,]),
    )
    assert "QE_forces" in si2.arrays
    assert si2.arrays["QE_forces"][0, 0] == approx(expected=-0.17099353, abs=1e-3)
    assert si2.arrays["QE_forces"][:, 1:] == approx(0.0)
    assert si2.arrays["QE_forces"][0] == approx(-1 * si2.arrays["QE_forces"][1])


def test_wfl_Espresso_calc(tmp_path, qe_cmd_and_pseudo):
    # command and pspot
    qe_cmd, pspot = qe_cmd_and_pseudo

    atoms = Atoms("Si", cell=(2, 2, 2), pbc=[True] * 3)
    kw = dict(
        pseudopotentials=dict(Si=os.path.basename(pspot)),
        input_data={"SYSTEM": {"ecutwfc": 40, "input_dft": "LDA",}},
        pseudo_dir=os.path.dirname(pspot),
        kpts=(2, 2, 2),
        conv_thr=0.0001
    ) 

    calc = wfl.calculators.espresso.Espresso(
        workdir=tmp_path,
        **kw)
    atoms.calc = calc

    atoms.get_potential_energy()
    atoms.get_forces()
    atoms.get_stress()


def test_wfl_Espresso_calc_via_generic(tmp_path, qe_cmd_and_pseudo):

    qe_cmd, pspot = qe_cmd_and_pseudo

    atoms = Atoms("Si", cell=(2, 2, 2), pbc=[True] * 3)
    kw = dict(
        pseudopotentials=dict(Si=os.path.basename(pspot)),
        input_data={"SYSTEM": {"ecutwfc": 40, "input_dft": "LDA",}},
        pseudo_dir=os.path.dirname(pspot),
        kpts=(2, 2, 2),
        conv_thr=0.0001,
        workdir=tmp_path
    ) 

    calc = (wfl.calculators.espresso.Espresso, [], kw)

    cfgs = [atoms]*3 + [Atoms("Cu", cell=(2, 2, 2), pbc=[True]*3)]
    ci = ConfigSet(cfgs)
    co = OutputSpec()
    autoparainfo = AutoparaInfo(
        num_python_subprocesses=0
    )

    ci = generic.calculate(
        inputs=ci,
        outputs=co, 
        calculator=calc, 
        output_prefix='qe_',
        autopara_info=autoparainfo
    )

    assert "qe_calculation_failed" in list(ci)[-1].info


