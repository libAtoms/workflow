from pathlib import Path
import shutil
import glob
import os
import copy

import numpy as np

import ase.io
import pytest
from ase.atoms import Atoms
from wfl.calculators.vasp import Vasp
from wfl.calculators import generic
from wfl.configset import ConfigSet, OutputSpec

test_vasp_mark = ('ASE_VASP_COMMAND' not in os.environ or
                  'ASE_VASP_COMMAND_GAMMA' not in os.environ or
                  'PYTEST_VASP_POTCAR_DIR' not in os.environ)
pytestmark = pytest.mark.skipif(test_vasp_mark, reason='missing env var ASE_VASP_COMMAND or ASE_VASP_COMMAND_GAMMA '
                                                       'or PYTEST_VASP_POTCAR_DIR')


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

    for run_dir in tmp_path.glob('run_VASP_*'):
        nfiles = len(list(os.scandir(run_dir)))

        assert nfiles == 18

        ats = list(configs_eval)
        assert 'TEST_energy' in ats[0].info
        assert 'TEST_forces' in ats[0].arrays
        # ase.io.write(sys.stdout, list(configs_eval), format='extxyz')
        with open(Path(run_dir) / "OUTCAR") as fin:
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

    for run_dir in tmp_path.glob('run_VASP_*'):
        nfiles = len(list(os.scandir(run_dir)))

        assert nfiles == 18

        ats = list(configs_eval)
        assert 'TEST_energy' in ats[0].info
        assert 'TEST_forces' in ats[0].arrays
        # ase.io.write(sys.stdout, list(configs_eval), format='extxyz')
        with open(Path(run_dir) / "OUTCAR") as fin:
            l = fin.readline()
            assert "gamma-only" in l


def test_vasp_gamma_auto(tmp_path):
    ase.io.write(tmp_path / 'vasp_in.xyz', Atoms('Si', cell=(6, 2, 2), pbc=[True] * 3), format='extxyz')

    # try with not large enough kspacing
    configs_eval = generic.calculate(
        inputs=ConfigSet(tmp_path / 'vasp_in.xyz'),
        outputs=OutputSpec('vasp_out.auto_gamma_no.xyz', file_root=tmp_path),
        calculator=Vasp(workdir=tmp_path, encut=200, pp=os.environ['PYTEST_VASP_POTCAR_DIR'],
                        keep_files=True, kspacing=2),
        output_prefix='TEST_',
    )

    for run_dir in tmp_path.glob('run_VASP_*'):
        nfiles = len(list(os.scandir(run_dir)))

        assert nfiles == 18

        for at in configs_eval:
            assert 'TEST_energy' in at.info
            assert 'TEST_forces' in at.arrays
        # ase.io.write(sys.stdout, list(configs_eval), format='extxyz')
        with open(Path(run_dir) / "OUTCAR") as fin:
            l = fin.readline()
            assert "gamma-only" not in l

        print("BOB clean non-gamma", run_dir)
        shutil.rmtree(run_dir)

    # try with large enough kspacing
    configs_eval = generic.calculate(
        inputs=ConfigSet(tmp_path / 'vasp_in.xyz'),
        outputs=OutputSpec('vasp_out.auto_gamma_yes.xyz', file_root=tmp_path),
        calculator=Vasp(workdir=tmp_path, encut=200, pp=os.environ['PYTEST_VASP_POTCAR_DIR'],
                        keep_files=True, kspacing=4),
        output_prefix='TEST_',
    )

    for run_dir in tmp_path.glob('run_VASP_*'):
        nfiles = len(list(os.scandir(run_dir)))

        assert nfiles == 18

        for at in configs_eval:
            assert 'TEST_energy' in at.info
            assert 'TEST_forces' in at.arrays
        # ase.io.write(sys.stdout, list(configs_eval), format='extxyz')
        with open(Path(run_dir) / "OUTCAR") as fin:
            l = fin.readline()
            assert "gamma-only" in l

        print("BOB clean gamma", run_dir)
        shutil.rmtree(run_dir)


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


def test_vasp_universal_kspacing(tmp_path):
    ats = [Atoms('Si', cell=(2, 2, 2), pbc=[True] * 3),
                  Atoms('Si', cell=(3, 3, 3), pbc=[True] * 3),
                  Atoms('Si', cell=(2, 2, 2), pbc=[True, True, False]),
                  Atoms('Si', cell=(2, 2, 2), pbc=[True, True, False])]
    ats[3].info["WFL_CALCULATOR_KWARGS"] = {"universal_kspacing": {"kspacing": 1.0, "kgamma": False}}
    ase.io.write(tmp_path / 'vasp_in.xyz', ats, format='extxyz')

    configs_eval = generic.calculate(
        inputs=ConfigSet(tmp_path / 'vasp_in.xyz'),
        outputs=OutputSpec('vasp_out.regular.xyz', file_root=tmp_path),
        calculator=(Vasp, [], {"workdir": tmp_path, "encut": 200, "universal_kspacing": {"kspacing": 1.0},
                               "pp": os.environ['PYTEST_VASP_POTCAR_DIR'], "keep_files": True}),
        output_prefix='TEST_')

    run_dirs = sorted(tmp_path.glob("run_VASP_*"), key=lambda f: f.stat().st_mtime)

    for run_dir, expected_mesh, expected_offset in zip(run_dirs,
                                                       [(4, 4, 4), (3, 3, 3), (4, 4, 1), (4, 4, 1)],
                                                       ["gamma"] * 3 + ["monkhorst-pack"]):
        nfiles = len(list(os.scandir(run_dir)))
        assert nfiles == 19
        with open(run_dir / "KPOINTS") as fin:
            l = fin.readlines()
            assert l[2].strip().lower() == expected_offset
            assert expected_mesh == tuple([int(f) for f in l[3].strip().split()])

    ats = list(configs_eval)
    for at in ats:
        assert 'TEST_energy' in at.info
        assert 'TEST_stress' in at.info
        assert 'TEST_forces' in at.arrays
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
    assert not scratch_dir.is_dir()

    ats = list(configs_eval)
    assert 'TEST_energy' in ats[0].info
    assert 'TEST_forces' in ats[0].arrays
    # ase.io.write(sys.stdout, list(configs_eval), format='extxyz')


def test_vasp_per_configuration(tmp_path):
    vasp_kwargs = {
        "encut": 200.0,  # kinetic energy cutoff
        "ediff": 1.0e-3,
        "kspacing": 1.0,
        "pp": os.environ['PYTEST_VASP_POTCAR_DIR'],
        "workdir": tmp_path
    }
    
    atoms = [Atoms('Si', cell=(2, 2, 2), pbc=[True] * 3),
             Atoms('Si', cell=(2, 2, 2), pbc=[True] * 3),
             Atoms('Si', cell=(2, 2, 2), pbc=[True] * 3)]

    tmp = copy.deepcopy(vasp_kwargs)
    tmp['encut'] = 220.0
    atoms[1].info["WFL_CALCULATOR_INITIALIZER"] = Vasp
    atoms[1].info["WFL_CALCULATOR_KWARGS"] = tmp

    tmp = copy.deepcopy(vasp_kwargs)
    tmp['encut'] = 240.0
    atoms[2].info["WFL_CALCULATOR_KWARGS"] = tmp

    calculator = (Vasp, [], vasp_kwargs)

    configs_eval = generic.calculate(
        inputs=ConfigSet(atoms),
        outputs=OutputSpec('vasp_out.regular.xyz', file_root=tmp_path),
        calculator=calculator,
        output_prefix='TEST_')
    
    ats = list(configs_eval)

    with open(tmp_path / ats[2].info['vasp_rundir'] / 'INCAR', 'r') as fincar:
        for l in fincar:
            if l.split('=')[0].strip() == 'ENCUT':
                assert float(l.split('=')[1]) == 240.0

    assert ats[0].info['TEST_energy'] > ats[1].info['TEST_energy'] > ats[2].info['TEST_energy']
    # ase.io.write(sys.stdout, list(configs_eval), format='extxyz')


def test_vasp_multi_calc(tmp_path):
    ase.io.write(tmp_path / 'vasp_in.xyz', Atoms('Si', cell=(2, 2, 2), pbc=[True] * 3), format='extxyz')

    configs_eval = generic.calculate(
        inputs=ConfigSet(tmp_path / 'vasp_in.xyz'),
        outputs=OutputSpec("vasp_out_ref.xyz", file_root=tmp_path),
        calculator=Vasp(workdir=tmp_path, encut=200, kspacing=1.0, pp=os.environ['PYTEST_VASP_POTCAR_DIR'],
                        keep_files=True),
        output_prefix='TEST_')
    e_GGA = list(configs_eval)[0].info["TEST_energy"]
    run_dir = list(tmp_path.glob('run_VASP_*'))[0]
    with open(run_dir / "OUTCAR") as fin:
        n_loop_GGA = sum(["LOOP" in l for l in fin])
    shutil.rmtree(run_dir)
    print("GGA E", e_GGA, "loops", n_loop_GGA)

    configs_eval = generic.calculate(
        inputs=ConfigSet(tmp_path / 'vasp_in.xyz'),
        outputs=OutputSpec('vasp_out.regular.xyz', file_root=tmp_path),
        calculator=Vasp(workdir=tmp_path, encut=200, kspacing=1.0, pp=os.environ['PYTEST_VASP_POTCAR_DIR'],
                        # keep_files=True, gga='PZ'),
                        keep_files=True, multi_calculation_kwargs=[{'lwave': True, 'gga': 'PZ'}, {}]),
        output_prefix='TEST_')
    e_test = list(configs_eval)[0].info["TEST_energy"]
    run_dir = list(tmp_path.glob('run_VASP_*'))[0]
    with open(run_dir / "OUTCAR") as fin:
        n_loop_test = sum(["LOOP" in l for l in fin])
    print("LDA + GGA E", e_test, "loops", n_loop_test)

    assert np.abs(e_GGA - e_test) < 1.0e-4
    assert n_loop_test < n_loop_GGA - 2


def test_vasp_unconverged(tmp_path):
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
    print("initial run converged?", ats[0].info["TEST_converged"])
    assert ats[0].info["TEST_converged"]


    configs_eval = generic.calculate(
        inputs=ConfigSet(tmp_path / 'vasp_in.xyz'),
        outputs=OutputSpec('vasp_out.unconverged.xyz', file_root=tmp_path),
        calculator=Vasp(workdir=tmp_path, encut=200, nelm=6, kspacing=1.0, pp=os.environ['PYTEST_VASP_POTCAR_DIR'],
                        keep_files=True),
        output_prefix='TEST_')

    run_dir = list(tmp_path.glob('run_VASP_*'))
    nfiles = len(list(os.scandir(run_dir[0])))

    assert nfiles == 18

    ats = list(configs_eval)
    print("second run converged?", ats[0].info["TEST_converged"])
    assert not ats[0].info["TEST_converged"]
