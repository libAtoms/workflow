from pathlib import Path
import glob
import os

import numpy as np

import ase.io
import pytest
from ase.atoms import Atoms
from wfl.calculators.vasp import Vasp
from wfl.calculators import generic
from wfl.configset import ConfigSet, OutputSpec

pytestmark = pytest.mark.skipif('ASE_VASP_COMMAND' not in os.environ or
                                'ASE_VASP_COMMAND_GAMMA' not in os.environ or
                                'PYTEST_VASP_POTCAR_DIR' not in os.environ,
                                reason='missing env var ASE_VASP_COMMAND or ASE_VASP_COMMAND_GAMMA or PYTEST_VASP_POTCAR_DIR')


def test_vasp_gamma(tmp_path, monkeypatch):
    ase.io.write(tmp_path / 'vasp_in.xyz', Atoms('Si', cell=(2, 2, 2), pbc=[False] * 3), format='extxyz')

    # try with env vars
    configs_eval = generic.calculate(
        inputs=ConfigSet(tmp_path / 'vasp_in.xyz'),
        outputs=OutputSpec('vasp_out.gamma.env.xyz', file_root=tmp_path),
        calculator=Vasp(workdir=tmp_path, encut=200, pp=os.environ['PYTEST_VASP_POTCAR_DIR'],
                        keep_files=True),
        output_prefix='TEST_', 
    )

    run_dir = list(tmp_path.glob('run_VASP_*'))
    nfiles = len(list(os.scandir(run_dir[0])))

    assert nfiles == 18

    ats = list(configs_eval)
    assert 'TEST_energy' in ats[0].info
    assert 'TEST_forces' in ats[0].arrays
    # ase.io.write(sys.stdout, list(configs_eval), format='extxyz')
    with open(Path(run_dir[0]) / "OUTCAR") as fin:
        l = fin.readline()
        assert "gamma-only" in l

    # try with command_gamma

    command_gamma = os.environ["ASE_VASP_COMMAND_GAMMA"]
    for cmd in Vasp.env_commands:
        monkeypatch.delenv(cmd + "_GAMMA", raising=False)
    configs_eval = generic.calculate(
        inputs=ConfigSet(tmp_path / 'vasp_in.xyz'),
        outputs=OutputSpec('vasp_out.gamma.command.xyz', file_root=tmp_path),
        calculator=Vasp(workdir=tmp_path, encut=200, pp=os.environ['PYTEST_VASP_POTCAR_DIR'],
                        keep_files=True, command_gamma=command_gamma),
        output_prefix='TEST_', 
    )

    run_dir = list(tmp_path.glob('run_VASP_*'))
    nfiles = len(list(os.scandir(run_dir[0])))

    assert nfiles == 18

    ats = list(configs_eval)
    assert 'TEST_energy' in ats[0].info
    assert 'TEST_forces' in ats[0].arrays
    # ase.io.write(sys.stdout, list(configs_eval), format='extxyz')
    with open(Path(run_dir[0]) / "OUTCAR") as fin:
        l = fin.readline()
        assert "gamma-only" in l


def test_vasp(tmp_path):
    ase.io.write(tmp_path / 'vasp_in.xyz', Atoms('Si', cell=(2, 2, 2), pbc=[True] * 3), format='extxyz')

    configs_eval = generic.calculate(
        inputs=ConfigSet(tmp_path / 'vasp_in.xyz'),
        outputs=OutputSpec('vasp_out.regular.xyz', file_root=tmp_path),
        calculator=Vasp(workdir=tmp_path, encut=200, kspacing=1.0, pp=os.environ['PYTEST_VASP_POTCAR_DIR'],
                        keep_files=True),
        output_prefix='TEST_')

    run_dir = list(tmp_path.glob('run_VASP_*'))
    nfiles = len(list(os.scandir(run_dir[0])))

    assert nfiles == 18

    ats = list(configs_eval)
    assert 'TEST_energy' in ats[0].info
    assert 'TEST_stress' in ats[0].info
    assert 'TEST_forces' in ats[0].arrays
    # import sys
    # ase.io.write(sys.stdout, list(configs_eval), format='extxyz')


def test_vasp_values(tmp_path):
    ats = Atoms(numbers=[14, 14], cell=(4, 2, 2), pbc=[True] * 3)
    ats.positions[1, 0] = 2.1
    ase.io.write(tmp_path / 'vasp_in.xyz', ats, format='extxyz')

    configs_eval = generic.calculate(
        inputs=ConfigSet(tmp_path / 'vasp_in.xyz'),
        outputs=OutputSpec('vasp_out.regular.xyz', file_root=tmp_path),
        calculator=Vasp(workdir=tmp_path, encut=200, kspacing=1.0, pp=os.environ['PYTEST_VASP_POTCAR_DIR'],
                        keep_files=True),
        output_prefix='TEST_')

    run_dir = list(tmp_path.glob('run_VASP_*'))
    nfiles = len(list(os.scandir(run_dir[0])))

    assert nfiles == 18

    ats = list(configs_eval)
    assert 'TEST_energy' in ats[0].info
    assert 'TEST_stress' in ats[0].info
    assert 'TEST_forces' in ats[0].arrays

    # import sys #NB
    # ase.io.write(sys.stdout, list(ats), format="extxyz") #NB

    assert np.allclose(ats[0].info["TEST_energy"], -0.55244948)
    assert np.allclose(ats[0].info["TEST_stress"], [-1.805493843235339, -1.7204368283214935, -1.7204368283214935, 0.0, 0.0, 0.0])
    assert np.allclose(ats[0].arrays["TEST_forces"], [[6.18419274, 0.0, 0.0], [-6.18419274, 0.0, 0.0]])


def test_vasp_negative_volume(tmp_path):
    at = Atoms('Si', cell=(2, 2, 2), pbc=[True] * 3)
    at.cell[0, :] = [0, 2, 0]
    at.cell[1, :] = [2, 0, 0]
    ase.io.write(tmp_path / 'vasp_in.xyz', at, format='extxyz')

    configs_eval = generic.calculate(
        inputs=ConfigSet(tmp_path / 'vasp_in.xyz'),
        outputs=OutputSpec('vasp_out.regular.xyz', file_root=tmp_path),
        calculator=Vasp(workdir=tmp_path, encut=200, kspacing=1.0, pp=os.environ['PYTEST_VASP_POTCAR_DIR'],
                        keep_files=True),
        output_prefix='TEST_')

    run_dir = list(tmp_path.glob('run_VASP_*'))
    nfiles = len(list(os.scandir(run_dir[0])))

    assert nfiles == 18

    ats = list(configs_eval)
    assert 'TEST_energy' in ats[0].info
    assert 'TEST_stress' in ats[0].info
    assert 'TEST_forces' in ats[0].arrays
    # ase.io.write(sys.stdout, list(configs_eval), format='extxyz')



def test_vasp_keep_default(tmp_path):
    ase.io.write(tmp_path / 'vasp_in.xyz', Atoms('Si', cell=(2, 2, 2), pbc=[True] * 3), format='extxyz')

    configs_eval = generic.calculate(
        inputs=ConfigSet(tmp_path / 'vasp_in.xyz'),
        outputs=OutputSpec('vasp_out.keep_default.xyz', file_root=tmp_path),
        calculator=Vasp(workdir=tmp_path, encut=200, kspacing=1.0, pp=os.environ['PYTEST_VASP_POTCAR_DIR'],
                             keep_files='default'),
        output_prefix='TEST_')

    run_dir = list(tmp_path.glob('run_VASP_*'))
    nfiles = len(list(os.scandir(run_dir[0])))

    assert nfiles == 5

    ats = list(configs_eval)
    assert 'TEST_energy' in ats[0].info
    assert 'TEST_stress' in ats[0].info
    assert 'TEST_forces' in ats[0].arrays
    # ase.io.write(sys.stdout, list(configs_eval), format='extxyz')


def test_vasp_keep_False(tmp_path):
    ase.io.write(tmp_path / 'vasp_in.xyz', Atoms('Si', cell=(2, 2, 2), pbc=[True] * 3), format='extxyz')

    configs_eval = generic.calculate(
        inputs=ConfigSet(tmp_path / 'vasp_in.xyz'),
        outputs=OutputSpec('vasp_out.keep_False.xyz', file_root=tmp_path),
        calculator=Vasp(workdir=tmp_path, encut=200, kspacing=1.0, pp=os.environ['PYTEST_VASP_POTCAR_DIR'],
                        keep_files=False),
        output_prefix='TEST_')

    run_dir = list(tmp_path.glob('run_VASP_*'))
    assert len(run_dir) == 0

    ats = list(configs_eval)
    assert 'TEST_energy' in ats[0].info
    assert 'TEST_stress' in ats[0].info
    assert 'TEST_forces' in ats[0].arrays
    # ase.io.write(sys.stdout, list(configs_eval), format='extxyz')


def test_vasp_to_SPC(tmp_path):
    ase.io.write(tmp_path / 'vasp_in.xyz', Atoms('Si', cell=(2, 2, 2), pbc=[True] * 3), format='extxyz')

    configs_eval = generic.calculate(
        inputs=ConfigSet(tmp_path / 'vasp_in.xyz'),
        outputs=OutputSpec('vasp_out.to_SPC.xyz', file_root=tmp_path),
        calculator=Vasp(workdir=tmp_path, encut=200, kspacing=1.0, pp=os.environ['PYTEST_VASP_POTCAR_DIR']),
        output_prefix=None)

    ats = list(configs_eval)
    assert 'energy' in ats[0].calc.results
    assert 'stress' in ats[0].calc.results
    assert 'forces' in ats[0].calc.results
    # ase.io.write(sys.stdout, list(configs_eval), format='extxyz')


def test_vasp_VASP_PP_PATH(tmp_path, monkeypatch):
    ase.io.write(tmp_path / 'vasp_in.xyz', Atoms('Si', cell=(2, 2, 2), pbc=[False] * 3), format='extxyz')

    monkeypatch.setenv("VASP_PP_PATH", os.environ['PYTEST_VASP_POTCAR_DIR'])
    configs_eval = generic.calculate(
        inputs=ConfigSet(tmp_path / 'vasp_in.xyz'),
        outputs=OutputSpec('vasp_out.gamma.xyz', file_root=tmp_path),
        calculator=Vasp(workdir=tmp_path, encut=200, keep_files=True),
        output_prefix='TEST_', 
    )

    run_dir = list(tmp_path.glob('run_VASP_*'))
    nfiles = len(list(os.scandir(run_dir[0])))

    assert nfiles == 18

    ats = list(configs_eval)
    assert 'TEST_energy' in ats[0].info
    assert 'TEST_forces' in ats[0].arrays
    # ase.io.write(sys.stdout, list(configs_eval), format='extxyz')


def test_vasp_scratchdir(tmp_path, monkeypatch):
    ase.io.write(tmp_path / 'vasp_in.xyz', Atoms('Si', cell=(2, 2, 2), pbc=[False] * 3), format='extxyz')

    monkeypatch.setenv("VASP_PP_PATH", os.environ['PYTEST_VASP_POTCAR_DIR'])
    configs_eval = generic.calculate(
        inputs=ConfigSet(tmp_path / 'vasp_in.xyz'),
        outputs=OutputSpec('vasp_out.gamma.xyz', file_root=tmp_path),
        calculator=Vasp(workdir=tmp_path, scratchdir="/tmp", encut=200, keep_files=True),
        output_prefix='TEST_', 
    )

    run_dir = list(tmp_path.glob('run_VASP_*'))
    nfiles = len(list(os.scandir(run_dir[0])))
    assert nfiles == 18

    scratch_dir = Path("/tmp") / str(run_dir[0].resolve()).replace("/", "", 1).replace("/", "_")
    assert not os.path.exists(scratch_dir)

    ats = list(configs_eval)
    assert 'TEST_energy' in ats[0].info
    assert 'TEST_forces' in ats[0].arrays
    # ase.io.write(sys.stdout, list(configs_eval), format='extxyz')
