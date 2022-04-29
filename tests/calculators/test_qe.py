"""
Tests for Quantum Espresso calculator interface
"""
import os
from shutil import which
import pytest

import ase.io
import numpy as np
import requests
from ase import Atoms
from ase.build import bulk
from pytest import approx, fixture, raises, skip

from wfl.calculators.espresso import evaluate_autopara_wrappable, qe_kpoints_and_kwargs
from wfl.calculators.dft import evaluate_dft
from wfl.configset import ConfigSet, OutputSpec


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

    # broken due to need for account/license click
    # url = "https://www.quantum-espresso.org/upf_files/Si.pbe-n-kjpaw_psl.1.0.0.UPF"
    url = "http://nninc.cnf.cornell.edu/psp_files/Si.pz-vbc.UPF"
    # alternative
    # url = "https://web.mit.edu/espresso_v6.1/amd64_linux26/qe-6.1/pseudo/Si.pz-vbc.UPF"

    # get the pseudo potential file, ~1.2MB
    try:
        r = requests.get(url)
    except requests.exceptions.ConnectionError:
        # no internet!
        skip(f"failed to make URL connection {url}")

    if r.status_code != requests.codes.ok:
        # the download has not worked
        skip(f"failed to download from URL {url}")

    # write to a temporary file
    pspot_file = tmp_path_factory.getbasetemp() / "Si.UPF"
    pspot_file.write_bytes(r.content)
    return cmd, pspot_file


def test_qe_kpoints():
    # PBC = TTT
    kw, prop = qe_kpoints_and_kwargs(
        Atoms("H", cell=[1, 1, 1], pbc=True), dict(kpts=(2, 3, 4)), ["energy", "stress"]
    )
    assert "tstress" in kw
    assert kw["tstress"]

    assert kw["kpts"] == (2, 3, 4)

    # PBC = FFF
    kw, prop = qe_kpoints_and_kwargs(
        Atoms("H", cell=[1, 1, 1], pbc=False),
        dict(kpts=(2, 3, 4)),
        ["energy", "stress", "forces"],
    )
    assert "tstress" in kw
    assert not kw["tstress"]

    assert "tprnfor" in kw
    assert kw["tprnfor"]

    assert "stress" not in prop
    assert kw["kpts"] is None
    assert kw["kspacing"] is None
    assert kw["koffset"] is False

    # PBC mixed -- kpts
    kw, prop = qe_kpoints_and_kwargs(
        Atoms("H", cell=[1, 1, 1], pbc=[True, False, False]),
        dict(kpts=(2, 3, 4), koffset=True),
        ["energy", "stress", "forces"],
    )

    assert "tstress" in kw
    assert not kw["tstress"]

    assert "tprnfor" in kw
    assert kw["tprnfor"]

    assert "stress" not in prop
    assert kw["kpts"] == (2, 1, 1)
    assert kw["kspacing"] is None
    assert kw["koffset"] == (1, 0, 0)

    # koffset in mixed PBC
    kw, prop = qe_kpoints_and_kwargs(
        Atoms("H", cell=[1, 1, 1], pbc=[True, False, False]),
        dict(kpts=(2, 3, 4), koffset=False),
        ["energy", "forces"],
    )
    assert kw["koffset"] is False

    # PBC mixed -- kspacing
    kw, prop = qe_kpoints_and_kwargs(
        Atoms("H", cell=[1, 1, 1], pbc=[True, False, False]),
        dict(kspacing=0.1, koffset=(0, 1, 0)),
        ["energy", "stress", "forces"],
    )
    assert "tstress" in kw
    assert not kw["tstress"]

    assert "tprnfor" in kw
    assert kw["tprnfor"]

    assert "stress" not in prop
    assert kw["kpts"] == (63, 1, 1)
    assert kw["kspacing"] is None
    assert kw["koffset"] == (0, 0, 0)


@pytest.mark.xfail(reason="PP file changes. Even before that hard-wired values are wrong, also calculation does not converge with default conv_thr")
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
        conv_thr=0.0001
    )

    # output container
    c_out = OutputSpec(
        file_root=tmp_path,
        output_files="qe_results.xyz",
        force=True,
        all_or_none=True,
    )

    results = evaluate_dft(
        calculator_name="QE",
        inputs=[at0, at],
        outputs=c_out,
        base_rundir=tmp_path,
        calculator_command=qe_cmd,
        calculator_kwargs=kw,
        output_prefix="QE_",
    )

    # unpack the configset
    si_single, si2 = list(results)

    # dev: type hints
    si_single: Atoms
    si2: Atoms

    # single atoms tests
    assert "QE_stress" not in si_single.info
    assert "QE_energy" in si_single.info
    assert si_single.info["QE_energy"] == approx(-601.8757092817176)
    assert si_single.get_volume() == approx(6.0 ** 3)

    # bulk Si tests
    assert "QE_energy" in si2.info
    assert si2.info["QE_energy"] == approx(-1214.4189734323988)
    assert "QE_stress" in si2.info
    assert si2.info["QE_stress"] == approx(
        abs=1e-5,
        expected=np.array([-0.040234, -0.040265, -0.040265, -0.002620, 0.0, 0.0]),
    )
    assert "QE_forces" in si2.arrays
    assert si2.arrays["QE_forces"][0, 0] == approx(-0.17277428)
    assert si2.arrays["QE_forces"][:, 1:] == approx(0.0)
    assert si2.arrays["QE_forces"][0] == approx(-1 * si2.arrays["QE_forces"][1])


def test_qe_errors():
    with raises(
        ValueError, match="QE will not perform a calculation without settings given!"
    ):
        evaluate_autopara_wrappable(Atoms(), calculator_kwargs=None)


def test_qe_no_calculation(tmp_path, qe_cmd_and_pseudo):
    # call just to skip if pw.x is missing
    _, _ = qe_cmd_and_pseudo

    results = evaluate_autopara_wrappable(bulk("Si"), calculator_kwargs=dict(), output_prefix="dummy_", base_rundir=tmp_path)

    assert isinstance(results, Atoms)
    assert "dummy_energy" not in results.info
    assert "dummy_stress" not in results.info
    assert "dummy_forces" not in results.arrays


def test_qe_to_spc(tmp_path, qe_cmd_and_pseudo):
    # command and pspot
    _, pspot = qe_cmd_and_pseudo

    # mainly a copy of VASP test
    ase.io.write(
        tmp_path / "qe_in.xyz",
        Atoms("Si", cell=(2, 2, 2), pbc=[True] * 3),
        format="extxyz",
    )

    kw = dict(
        pseudopotentials=dict(Si=os.path.basename(pspot)),
        input_data={"SYSTEM": {"ecutwfc": 40, "input_dft": "LDA",}},
        pseudo_dir=os.path.dirname(pspot),
        kpts=(2, 2, 2),
        conv_thr=0.0001
    )

    configs_eval = evaluate_dft(
        inputs=ConfigSet(input_files=tmp_path /  "qe_in.xyz"),
        outputs=OutputSpec(file_root=tmp_path, output_files="qe_out.to_SPC.xyz"),
        calculator_name="QE",
        base_rundir=tmp_path,
        calculator_kwargs=kw,
        output_prefix=None,
    )

    ats = list(configs_eval)
    assert "energy" in ats[0].calc.results
    assert "stress" in ats[0].calc.results
    assert "forces" in ats[0].calc.results
    # ase.io.write(sys.stdout, list(configs_eval), format='extxyz')
