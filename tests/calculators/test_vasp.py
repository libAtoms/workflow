import glob
import os

import ase.io
import pytest
from ase.atoms import Atoms
from wfl.calculators.dft import evaluate_dft
from wfl.configset import ConfigSet, OutputSpec

pytestmark = pytest.mark.skipif('VASP_COMMAND' not in os.environ or
                                'VASP_COMMAND_GAMMA' not in os.environ or
                                'TEST_VASP_POTCAR_DIR' not in os.environ,
                                reason='missing env var VASP_COMMAND or VASP_COMMAND_GAMMA or TEST_VASP_POTCAR_DIR')


def test_vasp_gamma(tmp_path):
    ase.io.write(os.path.join(tmp_path, 'vasp_in.xyz'), Atoms('Si', cell=(2, 2, 2), pbc=[False] * 3), format='extxyz')

    configs_eval = evaluate_dft(
        inputs=ConfigSet(input_files=os.path.join(tmp_path, 'vasp_in.xyz')),
        outputs=OutputSpec(file_root=tmp_path, output_files='vasp_out.gamma.xyz'),
        calculator_name="VASP",
        base_rundir=tmp_path,
        calculator_kwargs={'encut': 200, 'VASP_PP_PATH': os.environ['TEST_VASP_POTCAR_DIR']},
        output_prefix='TEST_',
        keep_files=True
    )

    run_dir = glob.glob(os.path.join(tmp_path, 'run_VASP_*'))
    nfiles = len([i for i in os.scandir(run_dir[0])])

    assert nfiles == 18

    ats = list(configs_eval)
    assert 'TEST_energy' in ats[0].info
    assert 'TEST_forces' in ats[0].arrays
    # ase.io.write(sys.stdout, list(configs_eval), format='extxyz')


def test_vasp(tmp_path):
    ase.io.write(os.path.join(tmp_path, 'vasp_in.xyz'), Atoms('Si', cell=(2, 2, 2), pbc=[True] * 3), format='extxyz')

    configs_eval = evaluate_dft(
        inputs=ConfigSet(input_files=os.path.join(tmp_path, 'vasp_in.xyz')),
        outputs=OutputSpec(file_root=tmp_path, output_files='vasp_out.regular.xyz'),
        calculator_name="VASP",
        base_rundir=tmp_path,
        calculator_kwargs={'encut': 200, 'kspacing': 1.0, 'VASP_PP_PATH': os.environ['TEST_VASP_POTCAR_DIR']},
        output_prefix='TEST_',
        keep_files=True)

    run_dir = glob.glob(os.path.join(tmp_path, 'run_VASP_*'))
    nfiles = len([i for i in os.scandir(run_dir[0])])

    assert nfiles == 18

    ats = list(configs_eval)
    assert 'TEST_energy' in ats[0].info
    assert 'TEST_stress' in ats[0].info
    assert 'TEST_forces' in ats[0].arrays
    # ase.io.write(sys.stdout, list(configs_eval), format='extxyz')


def test_vasp_keep_default(tmp_path):
    ase.io.write(os.path.join(tmp_path, 'vasp_in.xyz'), Atoms('Si', cell=(2, 2, 2), pbc=[True] * 3), format='extxyz')

    configs_eval = evaluate_dft(
        inputs=ConfigSet(input_files=os.path.join(tmp_path, 'vasp_in.xyz')),
        outputs=OutputSpec(file_root=tmp_path, output_files='vasp_out.keep_default.xyz'),
        calculator_name="VASP",
        base_rundir=tmp_path,
        calculator_kwargs={'encut': 200, 'kspacing': 1.0, 'VASP_PP_PATH': os.environ['TEST_VASP_POTCAR_DIR']},
        output_prefix='TEST_',
        keep_files='default')

    run_dir = glob.glob(os.path.join(tmp_path, 'run_VASP_*'))
    nfiles = len(list(os.scandir(run_dir[0])))

    assert nfiles == 5

    ats = list(configs_eval)
    assert 'TEST_energy' in ats[0].info
    assert 'TEST_stress' in ats[0].info
    assert 'TEST_forces' in ats[0].arrays
    # ase.io.write(sys.stdout, list(configs_eval), format='extxyz')


def test_vasp_keep_False(tmp_path):
    ase.io.write(os.path.join(tmp_path, 'vasp_in.xyz'), Atoms('Si', cell=(2, 2, 2), pbc=[True] * 3), format='extxyz')

    configs_eval = evaluate_dft(
        inputs=ConfigSet(input_files=os.path.join(tmp_path, 'vasp_in.xyz')),
        outputs=OutputSpec(file_root=tmp_path, output_files='vasp_out.keep_False.xyz'),
        calculator_name="VASP",
        base_rundir=tmp_path,
        calculator_kwargs={'encut': 200, 'kspacing': 1.0, 'VASP_PP_PATH': os.environ['TEST_VASP_POTCAR_DIR']},
        output_prefix='TEST_',
        keep_files=False)

    run_dir = glob.glob(os.path.join(tmp_path, 'run_VASP_*'))
    assert len(run_dir) == 0

    ats = list(configs_eval)
    assert 'TEST_energy' in ats[0].info
    assert 'TEST_stress' in ats[0].info
    assert 'TEST_forces' in ats[0].arrays
    # ase.io.write(sys.stdout, list(configs_eval), format='extxyz')


def test_vasp_to_SPC(tmp_path):
    ase.io.write(os.path.join(tmp_path, 'vasp_in.xyz'), Atoms('Si', cell=(2, 2, 2), pbc=[True] * 3), format='extxyz')

    configs_eval = evaluate_dft(
        inputs=ConfigSet(input_files=os.path.join(tmp_path, 'vasp_in.xyz')),
        outputs=OutputSpec(file_root=tmp_path, output_files='vasp_out.to_SPC.xyz'),
        calculator_name="VASP",
        base_rundir=tmp_path, calculator_kwargs={'encut': 200, 'kspacing': 1.0, 'VASP_PP_PATH': os.environ['TEST_VASP_POTCAR_DIR']},
        output_prefix=None)

    ats = list(configs_eval)
    assert 'energy' in ats[0].calc.results
    assert 'stress' in ats[0].calc.results
    assert 'forces' in ats[0].calc.results
    # ase.io.write(sys.stdout, list(configs_eval), format='extxyz')
