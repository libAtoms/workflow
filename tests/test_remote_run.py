import sys
import os
import shutil
import time
import json

from pathlib import Path
import numpy as np

import ase.io
from ase.atoms import Atoms

from ase.build import bulk
from ase.calculators.emt import EMT

import pytest

pytestmark = pytest.mark.remote

from wfl.configset import ConfigSet, OutputSpec
from wfl.calculators import generic
from wfl.generate import optimize, md
from wfl.calculators import generic
from wfl.calculators.vasp import Vasp
from wfl.calculators.espresso import Espresso
from wfl.autoparallelize import AutoparaInfo

from expyre.func import ExPyReJobDiedError

def test_generic_calc(tmp_path, expyre_systems, monkeypatch, remoteinfo_env):
    for sys_name in expyre_systems:
        if sys_name.startswith('_'):
            continue

        do_generic_calc(tmp_path, sys_name, monkeypatch, remoteinfo_env)


def test_generic_calc_qe(tmp_path, expyre_systems, monkeypatch, remoteinfo_env):
    for sys_name in expyre_systems:
        if sys_name.startswith('_'):
            continue

        do_generic_calc_qe(tmp_path, sys_name, monkeypatch, remoteinfo_env)


def test_minim(tmp_path, expyre_systems, monkeypatch, remoteinfo_env):
    for sys_name in expyre_systems:
        if sys_name.startswith('_'):
            continue

        do_minim(tmp_path, sys_name, monkeypatch, remoteinfo_env)


def test_md_deterministic(tmp_path, expyre_systems, monkeypatch, remoteinfo_env):
    for sys_name in expyre_systems:
        if sys_name.startswith('_'):
            continue

        do_md_deterministic(tmp_path, sys_name, monkeypatch, remoteinfo_env)


def test_vasp_fail(tmp_path, expyre_systems, monkeypatch, remoteinfo_env):
    for sys_name in expyre_systems:
        if sys_name.startswith('_'):
            continue

        do_vasp_fail(tmp_path, sys_name, monkeypatch, remoteinfo_env)


def do_vasp_fail(tmp_path, sys_name, monkeypatch, remoteinfo_env):
    ri = {'sys_name': sys_name, 'job_name': 'pytest_vasp_'+sys_name,
          'env_vars' : ['ASE_VASP_COMMAND=NONE', 'ASE_VASP_COMMAND_GAMMA=NONE'],
          'input_files' : ['POTCARs'],
          'resources': {'max_time': '5m', 'num_nodes': 1},
          'num_inputs_per_queued_job': 1, 'check_interval': 10}

    remoteinfo_env(ri)

    print('RemoteInfo', ri)

    nconfigs = 2
    nat = 8
    a0 = (nat * 20.0) ** (1.0/3.0)

    np.random.seed(5)
    ats = [Atoms(f'Al{nat}', cell=[a0]*3, scaled_positions=np.random.uniform(size=((nat, 3))), pbc=[True] * 3) for _ in range(nconfigs)]
    ase.io.write(tmp_path / f'ats_i_{sys_name}.xyz', ats)

    # calculate values via iterable loop with real jobs
    ci = ConfigSet(tmp_path / f'ats_i_{sys_name}.xyz')
    co = OutputSpec(tmp_path / f'ats_o_{sys_name}.xyz')

    # cd to test dir, so that things like relative paths for POTCARs work
    # NOTE: very cumbersome with VASP, maybe need more sophisticated control of staging files?
    monkeypatch.chdir(tmp_path)

    # create POTCAR
    (Path.cwd() / 'POTCARs' / 'Al').mkdir(parents=True, exist_ok=True)
    with open(Path.cwd() / 'POTCARs' / 'Al' / 'POTCAR', 'w') as fout:
        fout.write("\n")

    # jobs should fail because of bad executable
    results = generic.calculate(inputs=ci, outputs=co,
                          calculator=Vasp(encut= 200, pp='POTCARs'),
                          output_prefix='TEST_',
                          autopara_info={"remote_info": ri})

    for at in ase.io.read(tmp_path / f'ats_o_{sys_name}.xyz', ':'):
        ## ase.io.write(sys.stdout, at, format='extxyz')
        for k in at.info:
            if k == "TEST_calculation_failed":
                continue
            assert not k.startswith('TEST_')


def do_generic_calc(tmp_path, sys_name, monkeypatch, remoteinfo_env):
    ri = {'sys_name': sys_name, 'job_name': 'pytest_'+sys_name,
          'resources': {'max_time': '1h', 'num_nodes': 1},
          'num_inputs_per_queued_job': -36, 'check_interval': 10}

    remoteinfo_env(ri)

    print('RemoteInfo', ri)

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
    ci = ConfigSet(tmp_path / f'ats_i_{sys_name}.xyz')
    co = OutputSpec(tmp_path / f'ats_o_{sys_name}.xyz')

    # do not mark as processed so next call can reuse
    monkeypatch.setenv('WFL_EXPYRE_NO_MARK_PROCESSED', '1')

    t0 = time.time()
    results = generic.calculate(inputs=ci, outputs=co, calculator=(EMT, [], {}), properties=["energy", "forces"],
                          autopara_info={"remote_info": ri})
    dt = time.time() - t0
    print('remote parallel calc_time', dt)

    monkeypatch.delenv('WFL_EXPYRE_NO_MARK_PROCESSED')

    dev = [ (np.abs(at.info['EMT_energy'] - ref_E)) / np.maximum(np.abs(ref_E), 1.0e-3) for at, ref_E in zip(results, ref_Es) ]
    print('max deviation', max(dev))
    assert max(dev) < 1.0e-8

    # pretend to run again, as though it was interrupted
    (tmp_path / f'ats_o_{sys_name}.xyz').unlink()
    ci = ConfigSet(tmp_path / f'ats_i_{sys_name}.xyz')
    co = OutputSpec(tmp_path / f'ats_o_{sys_name}.xyz')

    t0 = time.time()
    results = generic.calculate(inputs=ci, outputs=co, calculator=(EMT, [], {}), properties=["energy", "forces"],
                          autopara_info={"remote_info": ri})
    dt_rerun = time.time() - t0
    print('remote parallel calc_time', dt_rerun)

    dev = [ (np.abs(at.info['EMT_energy'] - ref_E)) / np.maximum(np.abs(ref_E), 1.0e-3) for at, ref_E in zip(results, ref_Es) ]
    print('max deviation', max(dev))
    assert max(dev) < 1.0e-8

    # maybe can do the test without being so sensitive to timing?
    assert dt_rerun < dt / 4.0


# copied from calculators/test_qe.py::test_qe_calc
def do_generic_calc_qe(tmp_path, sys_name, monkeypatch, remoteinfo_env):
    ri = {'sys_name': sys_name, 'job_name': 'pytest_'+sys_name,
          'resources': {'max_time': '1h', 'num_nodes': 1},
          'num_inputs_per_queued_job': -36, 'check_interval': 10}

    pspot = tmp_path / "Si.UPF"
    shutil.copy(Path(__file__).parent / "assets" / "QE" / "Si.pz-vbc.UPF", pspot)

    remoteinfo_env(ri)
    print('RemoteInfo', ri)

    at = bulk("Si")
    at.positions[0, 0] += 0.01
    at0 = Atoms("Si", cell=[6.0, 6.0, 6.0], positions=[[3.0, 3.0, 3.0]], pbc=False)

    kw = dict(
        pseudopotentials=dict(Si=pspot.name),
        input_data={"SYSTEM": {"ecutwfc": 40, "input_dft": "LDA",}},
        kpts=(2, 2, 2),
        conv_thr=0.0001,
        pseudo_dir=str(pspot.parent)
    )

    calc = (Espresso, [], kw)

    # output container
    c_out = OutputSpec("qe_results.xyz", file_root=tmp_path)

    results = generic.calculate(
        inputs=[at0, at],
        outputs=c_out,
        calculator=calc,
        output_prefix='QE_',
        autopara_info={"remote_info": ri}
    )

    for at in results:
        assert "QE_energy" in at.info
        assert "QE_forces" in at.arrays


def do_minim(tmp_path, sys_name, monkeypatch, remoteinfo_env):
    ri = {'sys_name': sys_name, 'job_name': 'pytest_'+sys_name,
          'resources': {'max_time': '1h', 'num_nodes': 1},
          'num_inputs_per_queued_job': -36, 'check_interval': 10}

    remoteinfo_env(ri)

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
    ci = ConfigSet(infiles)

    # run locally
    co = OutputSpec([f.replace('_i_', '_o_local_') for f in infiles])
    results = optimize.optimize(inputs=ci, outputs=co, calculator=(EMT, [], {}), steps=5)

    co = OutputSpec([f.replace('_i_', '_o_') for f in infiles])
    t0 = time.time()
    results = optimize.optimize(inputs=ci, outputs=co, calculator=(EMT, [], {}), steps=5, autopara_info={"remote_info": ri})
    dt = time.time() - t0
    print('remote parallel calc_time', dt)

    # check consistency with local calc
    for at_local, at in zip(ase.io.read(tmp_path / f'ats_o_local_{sys_name}_1.xyz', ':'), ase.io.read(tmp_path / f'ats_o_{sys_name}_1.xyz', ':')):
        assert at_local.info['orig_file'] == at.info['orig_file']
        assert at_local.info['orig_file_seq_no'] == at.info['orig_file_seq_no']
        assert np.abs((at_local.info['last_op__optimize_energy'] - at.info['last_op__optimize_energy']) / at_local.info['last_op__optimize_energy']) < 1.0e-8
    for at_local, at in zip(ase.io.read(tmp_path / f'ats_o_local_{sys_name}_2.xyz', ':'), ase.io.read(tmp_path / f'ats_o_{sys_name}_2.xyz', ':')):
        assert at_local.info['orig_file'] == at.info['orig_file']
        assert at_local.info['orig_file_seq_no'] == at.info['orig_file_seq_no']
        assert np.abs((at_local.info['last_op__optimize_energy'] - at.info['last_op__optimize_energy']) / at_local.info['last_op__optimize_energy']) < 1.0e-8


def test_fail_immediately(tmp_path, expyre_systems, monkeypatch, remoteinfo_env):
    for sys_name in expyre_systems:
        if sys_name.startswith('_'):
            continue

        do_fail_immediately(tmp_path, sys_name, monkeypatch, remoteinfo_env)


def do_fail_immediately(tmp_path, sys_name, monkeypatch, remoteinfo_env):
    # make sure that by default correct exception is raised locally if it was raised by remote job
    ri = {'sys_name': sys_name, 'job_name': 'pytest_'+sys_name,
          'resources': {'max_time': '1m', 'num_nodes': 1},
          'num_inputs_per_queued_job': 1, 'check_interval': 10}
    remoteinfo_env(ri)

    ri = {"test_fail_immediately": ri}
    print("RemoteInfo", ri)

    ats = [Atoms('C') for _ in range(3)]
    ats[1].numbers = [14]

    calc = EMT()

    monkeypatch.setenv('WFL_EXPYRE_INFO', json.dumps(ri))
    with pytest.raises(NotImplementedError):
        results = generic.calculate(inputs=ConfigSet(ats), outputs=OutputSpec(), calculator=calc, properties=["energy", "forces"],
                              raise_calc_exceptions=True, autopara_info=AutoparaInfo(remote_label="test_fail_immediately"))


def test_ignore_failed_jobs(tmp_path, expyre_systems, monkeypatch, remoteinfo_env):
    for sys_name in expyre_systems:
        if sys_name.startswith('_'):
            continue

        do_ignore_failed_jobs(tmp_path, sys_name, monkeypatch, remoteinfo_env)

def do_ignore_failed_jobs(tmp_path, sys_name, monkeypatch, remoteinfo_env):
    # make sure that no exceptions are raised when ignore_failed_jobs=True is passed
    # even when remote job raises an exception
    ri = {'sys_name': sys_name, 'job_name': 'pytest_'+sys_name,
          'resources': {'max_time': '1m', 'num_nodes': 1},
          'num_inputs_per_queued_job': 1, 'check_interval': 10,
          'ignore_failed_jobs': True}
    remoteinfo_env(ri)

    ri = {"test_ignore_failed_jobs": ri}
    print("RemoteInfo", ri)

    ats = [Atoms('C') for _ in range(3)]
    ats[1].numbers = [14]

    calc = EMT()

    monkeypatch.setenv('WFL_EXPYRE_INFO', json.dumps(ri))
    results = generic.calculate(inputs=ConfigSet(ats), outputs=OutputSpec(), calculator=calc, properties=["energy", "forces"],
                          raise_calc_exceptions=True, autopara_info=AutoparaInfo(remote_label="test_ignore_failed_jobs"))


def test_resubmit_killed_jobs(tmp_path, expyre_systems, monkeypatch, remoteinfo_env):
    for sys_name in expyre_systems:
        if sys_name.startswith('_'):
            continue

        do_resubmit_killed_jobs(tmp_path, sys_name, monkeypatch, remoteinfo_env)

def do_resubmit_killed_jobs(tmp_path, sys_name, monkeypatch, remoteinfo_env):
    # make sure that jobs that time out can be resubmitted automatically
    ri = {'sys_name': sys_name, 'job_name': 'pytest_'+sys_name,
          'resources': {'max_time': '10s', 'num_nodes': 1},
          'num_inputs_per_queued_job': 1, 'check_interval': 10}
    remoteinfo_env(ri)

    print("RemoteInfo", ri)

    ats = [Atoms('C') for _ in range(3)]
    n = 40
    ats[1] = Atoms(f'C{n**3}', positions=np.asarray(np.meshgrid(range(n), range(n), range(n))).reshape((3, -1)).T,
                   cell=[n]*3, pbc=[True]*3)

    calc = (EMT, [], {})

    # first just ignore failures
    ri['ignore_failed_jobs'] = True
    ri['resubmit_killed_jobs'] = False
    monkeypatch.setenv('WFL_EXPYRE_INFO', json.dumps({"test_resubmit_killed_jobs": ri}))
    print("BOB ######### initial run, ignoring errors")
    results = generic.calculate(inputs=ConfigSet(ats), outputs=OutputSpec(), calculator=calc, properties=["energy", "forces"],
                          raise_calc_exceptions=True, autopara_info=AutoparaInfo(remote_label="test_resubmit_killed_jobs"))
    # make sure total number is correct, but only 2 have results
    assert len(list(results)) == len(ats)
    for at in results:
        print("BOB at.info", at.info.keys())
    assert sum(["EMT_energy" in at.info for at in results]) == len(ats) - 1

    # try with some failures that should result in ExPyReJobDied
    ri['ignore_failed_jobs'] = False
    ri['resubmit_killed_jobs'] = True
    monkeypatch.setenv('WFL_EXPYRE_INFO', json.dumps({"test_resubmit_killed_jobs": ri}))
    print("BOB ######### second run, should time out")
    with pytest.raises(ExPyReJobDiedError):
        results = generic.calculate(inputs=ConfigSet(ats), outputs=OutputSpec(), calculator=calc, properties=["energy", "forces"],
                              raise_calc_exceptions=True, autopara_info=AutoparaInfo(remote_label="test_resubmit_killed_jobs"))

    # now resubmit with longer time
    ri['resources']['max_time'] = '5m'
    monkeypatch.setenv('WFL_EXPYRE_INFO', json.dumps({"test_resubmit_killed_jobs": ri}))
    print("BOB ######### third run, should rerun 1 and succeed")
    # no easy way to check if one one has rerun, just check if all 3 succeeded this time
    results = generic.calculate(inputs=ConfigSet(ats), outputs=OutputSpec(), calculator=calc, properties=["energy", "forces"],
                          raise_calc_exceptions=True, autopara_info=AutoparaInfo(remote_label="test_resubmit_killed_jobs"))

    # make sure all have results
    assert len(list(results)) == len(ats)
    assert all(["EMT_energy" in at.info for at in results])


def do_md_deterministic(tmp_path, sys_name, monkeypatch, remoteinfo_env):
    ri = {'sys_name': sys_name, 'job_name': 'pytest_'+sys_name,
          'resources': {'max_time': '1h', 'num_nodes': 1},
          'num_inputs_per_queued_job': -36, 'check_interval': 10}

    remoteinfo_env(ri)

    print('RemoteInfo', ri)

    nconfigs = 40
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
    ci = ConfigSet(infiles)

    # run locally
    co = OutputSpec([f.replace('_i_', '_o_local_') for f in infiles])
    results_loc = md.md(inputs=ci, outputs=co, calculator=(EMT, [], {}), steps=10, dt=0.2, temperature=100.0, temperature_tau=10.0,
                        rng=np.random.default_rng(20))

    co = OutputSpec([f.replace('_i_', '_o_') for f in infiles])
    t0 = time.time()
    results_remote = md.md(inputs=ci, outputs=co, calculator=(EMT, [], {}), steps=10, dt=0.2, temperature=100.0, temperature_tau=10.0,
                           rng=np.random.default_rng(20), autopara_info={"remote_info": ri})
    dt = time.time() - t0
    print('remote parallel calc_time', dt)

    for at in zip(results_loc, results_remote):
        # print("BOB", [np.linalg.norm(loc.positions - remote.positions) < 1.0e-12 for (loc, remote) in zip(results_loc, results_remote)])
        assert all([np.linalg.norm(loc.positions - remote.positions) < 1.0e-12 for (loc, remote) in zip(results_loc, results_remote)])
