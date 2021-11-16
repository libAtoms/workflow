import sys
import os
import shutil
import time
import json

from pathlib import Path
import numpy as np

import ase.io
from ase.atoms import Atoms
from ase.calculators.emt import EMT

import pytest

pytestmark = pytest.mark.remote

from wfl.configset import ConfigSet_in, ConfigSet_out
from wfl.calculators import generic
from wfl.generate_configs import minim
from wfl.calculators.dft import evaluate_dft


def test_generic_calc(tmp_path, expyre_systems, monkeypatch):
    for sys_name in expyre_systems:
        if sys_name.startswith('_'):
            continue

        do_generic_calc(tmp_path, sys_name, monkeypatch)


# do we need this tested remotely as well?
def test_minim_local(tmp_path, expyre_systems, monkeypatch):
    for sys_name in expyre_systems:
        if sys_name.startswith('_'):
            continue

        do_minim(tmp_path, sys_name, monkeypatch)


def test_vasp_fail(tmp_path, expyre_systems, monkeypatch):
    for sys_name in expyre_systems:
        if sys_name.startswith('_'):
            continue

        do_vasp_fail(tmp_path, sys_name, monkeypatch)


def do_vasp_fail(tmp_path, sys_name, monkeypatch):
    ri = {'sys_name': sys_name, 'job_name': 'test_vasp_'+sys_name,
          'env_vars' : ['VASP_COMMAND=NONE', 'VASP_COMMAND_GAMMA=NONE'],
          'input_files' : ['POTCARs'],
          'resources': {'max_time': '5m', 'n': (1, 'nodes')},
          'job_chunksize': 1, 'check_interval': 10}

    if 'WFL_PYTEST_REMOTEINFO' in os.environ:
        ri_extra = json.loads(os.environ['WFL_PYTEST_REMOTEINFO'])
        ri.update(ri_extra)

    ri = {'test_remote_run.py::do_vasp_fail,calculators/dft.py::evaluate_dft': ri}
    print('RemoteInfo', ri)
    monkeypatch.setenv('WFL_AUTOPARA_REMOTEINFO', json.dumps(ri))

    nconfigs = 2
    nat = 8
    a0 = (nat * 20.0) ** (1.0/3.0)

    np.random.seed(5)
    ats = [Atoms(f'Al{nat}', cell=[a0]*3, scaled_positions=np.random.uniform(size=((nat, 3))), pbc=[True] * 3) for _ in range(nconfigs)]
    ase.io.write(tmp_path / f'ats_i_{sys_name}.xyz', ats)

    # calculate values via iterable loop with real jobs
    ci = ConfigSet_in(input_files=str(tmp_path / f'ats_i_{sys_name}.xyz'))
    co = ConfigSet_out(output_files=str(tmp_path / f'ats_o_{sys_name}.xyz'))

    # cd to test dir, so that things like relative paths for POTCARs work
    # NOTE: very cumbersome with VASP, maybe need more sophisticated control of staging files?
    monkeypatch.chdir(tmp_path)

    # create POTCAR
    (Path.cwd() / 'POTCARs' / 'Al').mkdir(parents=True, exist_ok=True)
    with open(Path.cwd() / 'POTCARs' / 'Al' / 'POTCAR', 'w') as fout:
        fout.write("\n")

    # jobs should fail because of bad executable
    results = evaluate_dft(inputs=ci, outputs=co, calculator_name='VASP',
                           base_rundir='.', calculator_kwargs={'encut': 200},
                           potcar_top_dir='POTCARs', output_prefix='TEST_')

    for at in ase.io.read(tmp_path / f'ats_o_{sys_name}.xyz', ':'):
        ## ase.io.write(sys.stdout, at, format='extxyz')
        for k in at.info:
            assert not k.startswith('TEST_')


def do_generic_calc(tmp_path, sys_name, monkeypatch):
    ri = {'sys_name': sys_name, 'job_name': 'test_'+sys_name,
          'resources': {'max_time': '1h', 'n': (1, 'nodes')},
          'job_chunksize': -36, 'check_interval': 10}

    if 'WFL_PYTEST_REMOTEINFO' in os.environ:
        ri_extra = json.loads(os.environ['WFL_PYTEST_REMOTEINFO'])
        if 'resources' in ri_extra:
            ri['resources'].update(ri_extra['resources'])
            del ri_extra['resources']
        ri.update(ri_extra)

    ri = {'test_remote_run.py::do_generic_calc,calculators/generic.py::run': ri}
    print('RemoteInfo', ri)
    monkeypatch.setenv('WFL_AUTOPARA_REMOTEINFO', json.dumps(ri))

    nconfigs = 1000
    nat = 40
    a0 = (nat * 20.0) ** (1.0/3.0)

    sys.stderr.write('Creating atoms\n')

    np.random.seed(5)
    ats = [Atoms(f'Al{nat}', cell=[a0]*3, scaled_positions=np.random.uniform(size=((nat, 3))), pbc=[True] * 3) for _ in range(nconfigs)]
    ase.io.write(tmp_path / f'ats_i_{sys_name}.xyz', ats)

    calc = EMT()

    # manually calculate reference values locally (in serial)
    t0 = time.time()
    sys.stderr.write('Calculating energies\n')
    ref_Es = []
    for at in ats:
        at.calc = calc
        ref_Es.append(at.get_potential_energy())
        at.calc = None
    dt = time.time() - t0
    print('local serial calc_time', dt)
    print('len(ref_Es)', len(ref_Es))

    # calculate values via iterable loop with real jobs
    ci = ConfigSet_in(input_files=str(tmp_path / f'ats_i_{sys_name}.xyz'))
    co = ConfigSet_out(output_files=str(tmp_path / f'ats_o_{sys_name}.xyz'))

    # do not mark as processed so next call can reuse
    monkeypatch.setenv('WFL_AUTOPARA_REMOTE_NO_MARK_PROCESSED', '1')

    t0 = time.time()
    results = generic.run(inputs=ci, outputs=co, calculator=calc)
    dt = time.time() - t0
    print('remote parallel calc_time', dt)

    monkeypatch.delenv('WFL_AUTOPARA_REMOTE_NO_MARK_PROCESSED')

    dev = [ (np.abs(at.info['EMT_energy'] - ref_E)) / np.maximum(np.abs(ref_E), 1.0e-3) for at, ref_E in zip(results, ref_Es) ]
    print('max deviation', max(dev))
    assert max(dev) < 1.0e-8

    # pretend to run again, as though it was interrupted
    (tmp_path / f'ats_o_{sys_name}.xyz').unlink()
    ci = ConfigSet_in(input_files=str(tmp_path / f'ats_i_{sys_name}.xyz'))
    co = ConfigSet_out(output_files=str(tmp_path / f'ats_o_{sys_name}.xyz'))

    t0 = time.time()
    results = generic.run(inputs=ci, outputs=co, calculator=calc)
    dt = time.time() - t0
    print('remote parallel calc_time', dt)

    dev = [ (np.abs(at.info['EMT_energy'] - ref_E)) / np.maximum(np.abs(ref_E), 1.0e-3) for at, ref_E in zip(results, ref_Es) ]
    print('max deviation', max(dev))
    assert max(dev) < 1.0e-8

    # maybe can do the test without being so sensitive to timing?
    assert dt < 20


def do_minim(tmp_path, sys_name, monkeypatch):
    ri = {'sys_name': sys_name, 'job_name': 'test_'+sys_name,
          'resources': {'max_time': '1h', 'n': (1, 'nodes')},
          'job_chunksize': -36, 'check_interval': 10}
    if 'WFL_PYTEST_REMOTEINFO' in os.environ:
        ri_extra = json.loads(os.environ['WFL_PYTEST_REMOTEINFO'])
        ri.update(ri_extra)

    ri = {'test_remote_run.py::do_minim,generate_configs/minim.py::run': ri}
    print('RemoteInfo', ri)

    nconfigs = 100
    nat = 40
    a0 = (nat * 20.0) ** (1.0/3.0)

    sys.stderr.write('Creating atoms\n')

    np.random.seed(5)
    ats_1 = [Atoms(f'Al{nat}', cell=[a0]*3, scaled_positions=np.random.uniform(size=((nat, 3))), pbc=[True] * 3) for _ in range(nconfigs // 2)]
    for at_i, at in enumerate(ats_1):
        at.info['orig_file'] = '1'
        at.info['orig_file_seq_no'] = at_i
    ats_2 = [Atoms(f'Al{nat}', cell=[a0]*3, scaled_positions=np.random.uniform(size=((nat, 3))), pbc=[True] * 3) for _ in range(nconfigs // 2)]
    for at_i, at in enumerate(ats_2):
        at.info['orig_file'] = '2'
        at.info['orig_file_seq_no'] = at_i
    ase.io.write(tmp_path / f'ats_i_{sys_name}_1.xyz', ats_1)
    ase.io.write(tmp_path / f'ats_i_{sys_name}_2.xyz', ats_2)

    infiles = [str(tmp_path / f'ats_i_{sys_name}_1.xyz'), str(tmp_path / f'ats_i_{sys_name}_2.xyz')]
    ci = ConfigSet_in(input_files=infiles)

    # run locally
    co = ConfigSet_out(output_files={f: f.replace('_i_', '_o_local_') for f in infiles})
    results = minim.run(inputs=ci, outputs=co, calculator=(EMT, [], {}), steps=5)

    # run remotely
    monkeypatch.setenv('WFL_AUTOPARA_REMOTEINFO', json.dumps(ri))

    co = ConfigSet_out(output_files={f: f.replace('_i_', '_o_') for f in infiles})
    t0 = time.time()
    results = minim.run(inputs=ci, outputs=co, calculator=(EMT, [], {}), steps=5)
    dt = time.time() - t0
    print('remote parallel calc_time', dt)

    # check consistency with local calc
    for at_local, at in zip(ase.io.read(tmp_path / f'ats_o_local_{sys_name}_1.xyz', ':'), ase.io.read(tmp_path / f'ats_o_{sys_name}_1.xyz', ':')):
        assert at_local.info['orig_file'] == at.info['orig_file']
        assert at_local.info['orig_file_seq_no'] == at.info['orig_file_seq_no']
        assert np.abs((at_local.info['minim_energy'] - at.info['minim_energy']) / at_local.info['minim_energy']) < 1.0e-8
    for at_local, at in zip(ase.io.read(tmp_path / f'ats_o_local_{sys_name}_2.xyz', ':'), ase.io.read(tmp_path / f'ats_o_{sys_name}_2.xyz', ':')):
        assert at_local.info['orig_file'] == at.info['orig_file']
        assert at_local.info['orig_file_seq_no'] == at.info['orig_file_seq_no']
        assert np.abs((at_local.info['minim_energy'] - at.info['minim_energy']) / at_local.info['minim_energy']) < 1.0e-8
