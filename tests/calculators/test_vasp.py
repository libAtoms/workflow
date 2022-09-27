from pathlib import Path
import glob
import os

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
    configs_eval = generic.run(
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
    configs_eval = generic.run(
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

    configs_eval = generic.run(
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


def test_vasp_negative_volume(tmp_path):
    at = Atoms('Si', cell=(2, 2, 2), pbc=[True] * 3)
    at.cell[0, :] = [0, 2, 0]
    at.cell[1, :] = [2, 0, 0]
    ase.io.write(tmp_path / 'vasp_in.xyz', at, format='extxyz')

    configs_eval = generic.run(
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

    configs_eval = generic.run(
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

    configs_eval = generic.run(
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

    configs_eval = generic.run(
        inputs=ConfigSet(tmp_path / 'vasp_in.xyz'),
        outputs=OutputSpec('vasp_out.to_SPC.xyz', file_root=tmp_path),
        calculator=Vasp(workdir=tmp_path, encut=200, kspacing=1.0, pp=os.environ['PYTEST_VASP_POTCAR_DIR']),
        output_prefix=None)

    ats = list(configs_eval)
    assert 'energy' in ats[0].calc.results
    assert 'stress' in ats[0].calc.results
    assert 'forces' in ats[0].calc.results
    # ase.io.write(sys.stdout, list(configs_eval), format='extxyz')

####################################################################################################

def test_vasp_VASP_PP_PATH(tmp_path, monkeypatch):
    ase.io.write(tmp_path / 'vasp_in.xyz', Atoms('Si', cell=(2, 2, 2), pbc=[False] * 3), format='extxyz')

    monkeypatch.setenv("VASP_PP_PATH", os.environ['PYTEST_VASP_POTCAR_DIR'])
    configs_eval = generic.run(
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

def test_vasp_reuse_rundir(tmp_path, monkeypatch):
    ase.io.write(tmp_path / 'vasp_in.xyz', Atoms('Si', cell=(2, 2, 2), pbc=[False] * 3), format='extxyz')

    monkeypatch.setenv("VASP_PP_PATH", os.environ['PYTEST_VASP_POTCAR_DIR'])
    configs_eval = generic.run(
        inputs=ConfigSet(tmp_path / 'vasp_in.xyz'),
        outputs=OutputSpec('vasp_out.gamma.xyz', file_root=tmp_path),
        calculator=Vasp(workdir=tmp_path, rundir="run_VASP", reuse_rundir=True, encut=200, keep_files=True),
        output_prefix='TEST_', 
    )

    run_dir = list(tmp_path.glob('run_VASP'))
    nfiles = len(list(os.scandir(run_dir[0])))

    assert nfiles == 18

    ats = list(configs_eval)
    assert 'TEST_energy' in ats[0].info
    assert 'TEST_forces' in ats[0].arrays
    # ase.io.write(sys.stdout, list(configs_eval), format='extxyz')


def test_vasp_scratchdir(tmp_path, monkeypatch):
    ase.io.write(tmp_path / 'vasp_in.xyz', Atoms('Si', cell=(2, 2, 2), pbc=[False] * 3), format='extxyz')

    monkeypatch.setenv("VASP_PP_PATH", os.environ['PYTEST_VASP_POTCAR_DIR'])
    configs_eval = generic.run(
        inputs=ConfigSet(tmp_path / 'vasp_in.xyz'),
        outputs=OutputSpec('vasp_out.gamma.xyz', file_root=tmp_path),
        calculator=Vasp(workdir=tmp_path, scratchdir="/tmp", encut=200, keep_files=True),
        output_prefix='TEST_', 
    )

    run_dir = list(tmp_path.glob('run_VASP_*'))
    nfiles = len(list(os.scandir(run_dir[0])))
    assert nfiles == 18

    scratch_dir = Path("/tmp") / str(run_dir[0].resolve()).replace("/", "", 1)
    assert os.path.exists(scratch_dir)
    nfiles_scratch = len(list(os.scandir(scratch_dir)))
    assert nfiles_scratch == 0

    ats = list(configs_eval)
    assert 'TEST_energy' in ats[0].info
    assert 'TEST_forces' in ats[0].arrays
    # ase.io.write(sys.stdout, list(configs_eval), format='extxyz')
