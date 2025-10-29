from pytest import approx
import pytest

import os
from pathlib import Path
import json

import numpy as np

from ase import Atoms
import ase.io
from ase.build import bulk
from ase.calculators.emt import EMT
from ase.units import fs
from ase.md.logger import MDLogger
from wfl.autoparallelize import autoparainfo

from wfl.generate import md
from wfl.configset import ConfigSet, OutputSpec
from wfl.generate.md.abort import AbortOnCollision, AbortOnLowEnergy

try:
    from ase.md.langevinbaoab import LangevinBAOAB
except ImportError:
    LangevinBAOAB = None

def select_every_10_steps_for_tests_during(at):
    return at.info.get("MD_step", 1) % 10 == 0

def select_every_10_steps_for_tests_after(traj):
    return [at for at in traj if at.info["MD_step"] % 10 == 0]

def check_validity_for_tests(at):
    if "5" in str(at.info["MD_step"]):
        return False
    return True

@pytest.fixture
def cu_slab():

    atoms = bulk("Cu", "fcc", a=3.6, cubic=True)
    atoms *= (2, 2, 2)
    atoms.rattle(stdev=0.01, seed=159)

    atoms.info['config_type'] = 'cu_slab'
    atoms.info['buildcell_config_i'] = 'fake_buildecell_config_name'

    return atoms


def test_NVE(cu_slab):
    calc = EMT()

    inputs = ConfigSet(cu_slab)
    outputs = OutputSpec()

    atoms_traj = md.md(inputs, outputs, calculator=calc, steps=300, dt=1.0,
                       temperature=500.0, rng=np.random.default_rng(1))

    atoms_traj = list(atoms_traj)

    assert len(atoms_traj) == 301


def test_NVT_const_T(cu_slab):

    calc = EMT()

    inputs = ConfigSet(cu_slab)
    outputs = OutputSpec()

    atoms_traj = md.md(inputs, outputs, calculator=calc, steps=300, dt=1.0,
                       temperature=500.0, temperature_tau=30.0, rng=np.random.default_rng(1))

    atoms_traj = list(atoms_traj)

    assert len(atoms_traj) == 301
    assert all([at.info['MD_temperature_K'] == 500.0 for at in atoms_traj])
    assert np.all(atoms_traj[0].cell == atoms_traj[-1].cell)


def test_NVT_Langevin_const_T(cu_slab):
    calc = EMT()

    inputs = ConfigSet(cu_slab)
    outputs = OutputSpec()

    atoms_traj = md.md(inputs, outputs, calculator=calc, integrator="Langevin", steps=300, dt=1.0,
                       temperature=500.0, temperature_tau=100/fs, rng=np.random.default_rng(1))

    atoms_traj = list(atoms_traj)

    assert len(atoms_traj) == 301
    assert all([at.info['MD_temperature_K'] == 500.0 for at in atoms_traj])
    assert np.all(atoms_traj[0].cell == atoms_traj[-1].cell)


def test_NPT_Langevin_fail(cu_slab):
    calc = EMT()

    inputs = ConfigSet(cu_slab)
    outputs = OutputSpec()

    with pytest.raises(ValueError):
        atoms_traj = md.md(inputs, outputs, calculator=calc, integrator="Langevin", steps=300, dt=1.0,
                           temperature=500.0, temperature_tau=100/fs, pressure=0.0,
                           rng=np.random.default_rng(1))


def test_NPT_Berendsen_hydro_F_fail(cu_slab):
    calc = EMT()

    inputs = ConfigSet(cu_slab)
    outputs = OutputSpec()

    with pytest.raises(ValueError):
        atoms_traj = md.md(inputs, outputs, calculator=calc, integrator="Berendsen", steps=300, dt=1.0,
                           temperature=500.0, temperature_tau=100/fs, pressure=0.0, hydrostatic=False,
                           rng=np.random.default_rng(1))


def test_NPT_Berendsen_NPH_fail(cu_slab):
    calc = EMT()

    inputs = ConfigSet(cu_slab)
    outputs = OutputSpec()

    with pytest.raises(ValueError):
        atoms_traj = md.md(inputs, outputs, calculator=calc, integrator="Berendsen", steps=300, dt=1.0,
                           pressure=0.0,
                           rng=np.random.default_rng(1))


def test_NPT_Berendsen(cu_slab):
    calc = EMT()

    inputs = ConfigSet(cu_slab)
    outputs = OutputSpec()

    atoms_traj = md.md(inputs, outputs, calculator=calc, integrator="Berendsen", steps=300, dt=1.0,
                       temperature=500.0, temperature_tau=100/fs, pressure=0.0,
                       rng=np.random.default_rng(1))

    atoms_traj = list(atoms_traj)
    print("I cell", atoms_traj[0].cell)
    print("F cell", atoms_traj[1].cell)

    assert len(atoms_traj) == 301
    assert all([at.info['MD_temperature_K'] == 500.0 for at in atoms_traj])
    assert np.any(atoms_traj[0].cell != atoms_traj[-1].cell)

    cell_f = atoms_traj[0].cell[0, 0] / atoms_traj[-1].cell[0, 0]
    assert np.allclose(atoms_traj[0].cell, atoms_traj[-1].cell * cell_f)


@pytest.mark.skipif(LangevinBAOAB is None, reason="No LangevinBAOAB available")
def test_NPT_LangevinBAOAB(cu_slab):
    calc = EMT()

    inputs = ConfigSet(cu_slab)
    outputs = OutputSpec()

    atoms_traj = md.md(inputs, outputs, calculator=calc, integrator="LangevinBAOAB", steps=300, dt=1.0,
                       temperature=500.0, temperature_tau=100/fs, pressure=0.0,
                       rng=np.random.default_rng(1))

    atoms_traj = list(atoms_traj)
    print("I cell", atoms_traj[0].cell)
    print("F cell", atoms_traj[1].cell)

    assert len(atoms_traj) == 301
    assert all([at.info['MD_temperature_K'] == 500.0 for at in atoms_traj])
    assert np.any(atoms_traj[0].cell != atoms_traj[-1].cell)

    cell_f = atoms_traj[0].cell[0, 0] / atoms_traj[-1].cell[0, 0]
    assert np.allclose(atoms_traj[0].cell, atoms_traj[-1].cell * cell_f)


@pytest.mark.skipif(LangevinBAOAB is None, reason="No LangevinBAOAB available")
def test_NPT_LangevinBAOAB_hydro_F(cu_slab):
    calc = EMT()

    inputs = ConfigSet(cu_slab)
    outputs = OutputSpec()

    atoms_traj = md.md(inputs, outputs, calculator=calc, integrator="LangevinBAOAB", steps=300, dt=1.0,
                       temperature=500.0, temperature_tau=100/fs, pressure=0.0, hydrostatic=False,
                       rng=np.random.default_rng(1))

    atoms_traj = list(atoms_traj)
    print("I cell", atoms_traj[0].cell)
    print("F cell", atoms_traj[1].cell)

    assert len(atoms_traj) == 301
    assert all([at.info['MD_temperature_K'] == 500.0 for at in atoms_traj])
    assert np.any(atoms_traj[0].cell != atoms_traj[-1].cell)

    cell_f = atoms_traj[0].cell[0, 0] / atoms_traj[-1].cell[0, 0]
    assert not np.allclose(atoms_traj[0].cell, atoms_traj[-1].cell * cell_f)



def test_NVT_Langevin_const_T_per_config(cu_slab):

    calc = EMT()

    inputs = ConfigSet([cu_slab.copy(), cu_slab.copy()])
    outputs = OutputSpec()

    for at_i, at in enumerate(inputs):
        at.info["WFL_MD_KWARGS"] = json.dumps({'temperature': 500 + at_i * 100})

    n_steps = 30

    atoms_traj = md.md(inputs, outputs, calculator=calc, integrator="Langevin", steps=n_steps, dt=1.0,
                       temperature=200.0, temperature_tau=100/fs, rng=np.random.default_rng(1))

    atoms_traj = list(atoms_traj)
    atoms_final = atoms_traj[-1]

    assert len(atoms_traj) == (n_steps + 1) * 2
    assert all([at.info['MD_temperature_K'] == 500.0 for at in list(atoms_traj)[:n_steps + 1]])
    assert all([at.info['MD_temperature_K'] == 600.0 for at in list(atoms_traj)[n_steps + 1:]])


def test_NVT_const_T_mult_configs_distinct_seeds(cu_slab):

    calc = EMT()

    inputs = ConfigSet([cu_slab.copy() for _ in range(4)])
    outputs = OutputSpec()

    atoms_traj = md.md(inputs, outputs, calculator=calc, steps=300, dt=1.0,
                       temperature=500.0, temperature_tau=30.0, rng=np.random.default_rng(23875))

    last_configs = [list(group)[-1] for group in atoms_traj.groups()]
    last_vs = [np.linalg.norm(at.get_velocities()) for at in last_configs]
    assert all([v != last_vs[0] for v in last_vs[1:]])


def test_NVT_simple_ramp(cu_slab):

    calc = EMT()

    inputs = ConfigSet(cu_slab)
    outputs = OutputSpec()

    atoms_traj = md.md(inputs, outputs, calculator=calc, steps=300, dt=1.0,
                       temperature=(500.0, 100.0), temperature_tau=30.0, rng=np.random.default_rng(1))

    atoms_traj = list(atoms_traj)
    atoms_final = atoms_traj[-1]

    assert len(atoms_traj) == 301
    Ts = []
    for T_i, T in enumerate(np.linspace(500.0, 100.0, 10)):
        Ts.extend([T] * 30)
        if T_i == 0:
            Ts.append(T)
    assert all(np.isclose(Ts, [at.info['MD_temperature_K'] for at in atoms_traj]))


def test_NVT_complex_ramp(cu_slab):

    calc = EMT()

    inputs = ConfigSet(cu_slab)
    outputs = OutputSpec()

    atoms_traj = md.md(inputs, outputs, calculator=calc, steps=300, dt=1.0,
                           temperature=[{'T_i': 100.0, 'T_f': 500.0, 'traj_frac': 0.5},
                                        {'T_i': 500.0, 'T_f': 500.0, 'traj_frac': 0.25},
                                        {'T_i': 500.0, 'T_f': 300.0, 'traj_frac': 0.25}],
                           temperature_tau=30.0, rng=np.random.default_rng(1))

    atoms_traj = list(atoms_traj)
    atoms_final = atoms_traj[-1]

    assert len(atoms_traj) == 306
    Ts = []
    for T_i, T in enumerate(np.linspace(100.0, 500.0, 10)):
        Ts.extend([T] * 15)
        if T_i == 0:
            Ts.append(T)
    Ts.extend([500.0] * 75)
    for T_i, T in enumerate(np.linspace(500.0, 300.0, 10)):
        Ts.extend([T] * 8)

    # for at_i, at in enumerate(atoms_traj):
        # print(at_i, at.info['MD_time_fs'], 'MD', at.info['MD_temperature_K'], 'test', Ts[at_i])

    assert all(np.isclose(Ts, [at.info['MD_temperature_K'] for at in atoms_traj]))


def test_subselector_function_after(cu_slab):

    calc = EMT()

    inputs = ConfigSet(cu_slab)
    outputs = OutputSpec()

    atoms_traj = md.md(inputs, outputs, calculator=calc, steps=300, dt=1.0,
                       temperature=500.0, traj_select_after_func=select_every_10_steps_for_tests_after,
                       rng=np.random.default_rng(1))

    atoms_traj = list(atoms_traj)
    assert len(atoms_traj) == 31


def test_subselector_function_during(cu_slab):

    calc = EMT()

    for steps in [300, 301]:
        inputs = ConfigSet(cu_slab)
        outputs = OutputSpec()

        atoms_traj = md.md(inputs, outputs, calculator=calc, steps=steps, dt=1.0,
                           temperature=500.0, traj_select_during_func=select_every_10_steps_for_tests_during,
                           rng=np.random.default_rng(1))

        atoms_traj = list(atoms_traj)
        assert len(atoms_traj) == 31


def test_md_abort_function(cu_slab):

    calc = EMT()
    autopara_info = autoparainfo.AutoparaInfo(skip_failed=False)

    # test bulk Cu slab for collision
    inputs = ConfigSet(cu_slab)
    outputs = OutputSpec()
    md_stopper = AbortOnCollision(collision_radius=2.25)

    # why doesn't this throw an raise a RuntimeError even if md failed and `skip_failed` is False?
    atoms_traj = md.md(inputs, outputs, calculator=calc, steps=500, dt=10.0,
                       temperature=2000.0, abort_check=md_stopper, autopara_info=autopara_info,
                       rng=np.random.default_rng(1))

    assert len(list(atoms_traj)) < 501

    # test surface Cu slab for low energy
    cu_slab.set_cell(cu_slab.cell * (1, 1, 2), False)
    inputs = ConfigSet(cu_slab)
    outputs = OutputSpec()

    md_stopper = AbortOnLowEnergy(0.0007)
    atoms_traj = md.md(inputs, outputs, calculator=calc, steps=500, dt=10.0,
                       temperature=5.0, abort_check=md_stopper, autopara_info=autopara_info,
                       rng=np.random.default_rng(1))

    assert len(list(atoms_traj)) < 501


def test_md_attach_logger(cu_slab, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    calc = EMT()
    autopara_info = autoparainfo.AutoparaInfo(num_python_subprocesses=2, num_inputs_per_python_subprocess=1, skip_failed=False)

    inputs = ConfigSet([cu_slab, cu_slab])
    outputs = OutputSpec()

    logger_kwargs = {
        "logger" : MDLogger,
        "logfile" : "test_log",
    }

    atoms_traj = md.md(inputs, outputs, calculator=calc, integrator="Langevin", steps=300, dt=1.0,
                           temperature=500.0, temperature_tau=100/fs, logger_kwargs=logger_kwargs, logger_interval=1,
                           rng=np.random.default_rng(1), autopara_info=autopara_info,)

    atoms_traj = list(atoms_traj)
    atoms_final = atoms_traj[-1]

    workdir = Path(os.getcwd())

    assert len(atoms_traj) == 602
    assert all([Path(workdir / "test_log.config_0").is_file(), Path(workdir / "test_log.config_1").is_file()])


def test_md_attach_logger_stdout(cu_slab, tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)

    calc = EMT()
    autopara_info = autoparainfo.AutoparaInfo(num_python_subprocesses=2, num_inputs_per_python_subprocess=1, skip_failed=False)

    inputs = ConfigSet([cu_slab, cu_slab])
    outputs = OutputSpec()

    logger_kwargs = {
        "logger" : MDLogger,
        "logfile" : "-",
    }

    atoms_traj = md.md(inputs, outputs, calculator=calc, integrator="Langevin", steps=300, dt=1.0,
                           temperature=500.0, temperature_tau=100/fs, logger_kwargs=logger_kwargs, logger_interval=1,
                           rng=np.random.default_rng(1), autopara_info=autopara_info,)

    atoms_traj = list(atoms_traj)
    atoms_final = atoms_traj[-1]

    workdir = Path(os.getcwd())

    assert len(atoms_traj) == 602

    # make sure normal log files were not written
    assert len(list(Path(workdir).glob("*"))) == 0

    captured = capsys.readouterr()
    n_0 = sum(['item 0 ' in captured.out.splitlines()])
    n_1 = sum(['item 1 ' in captured.out.splitlines()])
    if n_0 != 301 or n_1 != 301:
        pytest.xfail("capsys fails to capture stdout to check for logger output")
