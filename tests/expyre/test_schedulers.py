import sys
import os
import time
import json

import pytest
pytestmark = pytest.mark.remote

from pathlib import Path

from wfl.expyre.subprocess import subprocess_run, subprocess_copy
from wfl.expyre.resources import Resources


def test_working_job(tmp_path, expyre_config):
    from wfl.expyre.config import systems

    for sys_name, system in systems.items():
        if sys_name.startswith('_'):
            continue

        remote_rundir = str(Path(systems[sys_name].remote_rundir) / f'stage_dummy_working_{sys_name}')
        subprocess_run(system.host, ['rm', '-r', f'{remote_rundir}', ';', 'mkdir', '-p', f'{remote_rundir}'],
                       remsh_cmd=system.remsh_cmd)

        sys.stderr.write(f'Testing working {sys_name} remote_rundir {remote_rundir}\n')
        do_working_job(system, remote_rundir, tmp_path, system.host)


def test_definitely_queued_job(tmp_path, expyre_config):
    if 'EXPYRE_PYTEST_QUEUED_JOB_RESOURCES' not in os.environ:
        pytest.xfail('test_definitely_queued_job() needs EXPYRE_PYTEST_QUEUED_JOB_RESOURCES '
                     'env var containing resources for job that uses all resources, so it '
                     'will definitely get stuck in queued state (see README)')

    from wfl.expyre.config import systems

    did_work=False
    for sys_name, system in systems.items():
        if sys_name.startswith('_'):
            continue

        try:
            queued_job_resources = json.loads(os.environ['EXPYRE_PYTEST_QUEUED_JOB_RESOURCES'])
        except json.decoder.JSONDecodeError:
            with open(os.environ['EXPYRE_PYTEST_QUEUED_JOB_RESOURCES']) as fin:
                queued_job_resources = json.load(fin)

        if sys_name not in queued_job_resources:
            continue

        assert len(queued_job_resources[sys_name]) == 2

        did_work = True
        do_definitely_queued_job(tmp_path, expyre_config, sys_name, queued_job_resources[sys_name])

    if not did_work:
        pytest.skip('test_definitely_queued_job found EXPYRE_PYTEST_QUEUED_JOB_RESOURCES but no system matched')


def do_working_job(system, remote_rundir, tmp_path, copy_back_host):
    r = Resources(n=(2, 'nodes'), max_time='5m')
    partition, node_dict = r.find_nodes(system.partitions)

    sched = system.scheduler
    remote_job = sched.submit(id='dummy_id_1', remote_dir=remote_rundir, partition=partition,
                              commands=['echo STARTING',
                                        'pwd | sed "s#^${HOME}/##"',
                                        'echo BOB',
                                        'sleep 30',
                                        'echo ENDING'],
                              max_time=r.max_time,
                              header=system.queuing_sys_header, node_dict=node_dict)
    assert isinstance(remote_job, str) and len(remote_job) > 0

    # wait for status not queued or running
    time.sleep(5)
    status = sched.status(remote_job)
    print('got status', status)
    while status[remote_job] in ['queued', 'running']:
        time.sleep(5)
        status = sched.status(remote_job)
        print('got status', status)

    assert status[remote_job] == 'done'

    if copy_back_host is not None:
        subprocess_copy(remote_rundir + '/job.dummy_id_1.stdout', Path(tmp_path), from_host=copy_back_host, remsh_cmd=system.remsh_cmd)
        local_stdout = tmp_path / 'job.dummy_id_1.stdout'
    else:
        local_stdout = remote_rundir + '/job.dummy_id_1.stdout'

    # check stdout content
    with open(local_stdout) as fin:
        lines = []
        for l in fin.readlines():
            if l.strip == 'STARTING':
                lines.append(l)
            elif l.strip == 'ENDING':
                break
    lines = lines[1:]
    remote_rundir_comparison = remote_rundir.replace(os.environ['HOME']+'/', '')
    assert all([l1.strip() == l2.strip() for l1, l2 in zip(lines, [remote_rundir_comparison, 'BOB'])])


def do_definitely_queued_job(tmp_path, expyre_config, sys_name, queued_job_resources):
    from wfl.expyre.config import systems

    system = systems[sys_name]
    sys.stderr.write(f'Testing definitely queued {sys_name}\n')

    sched = system.scheduler

    # one job to take space, queued or running
    system.run(['mkdir', '-p', system.remote_rundir + '/stage_dummy_queued_2'])

    queued_job_resources[0]['max_time'] = '5m'
    r = Resources(**(queued_job_resources[0]))
    partition, node_dict = r.find_nodes(system.partitions)
    remote_job_0 = sched.submit(id='dummy_id_2', remote_dir=system.remote_rundir + '/stage_dummy_queued_2', partition=partition,
                              commands=['pwd', 'echo BOB', 'sleep 120'], max_time=r.max_time,
                              header=system.queuing_sys_header, node_dict=node_dict)
    assert isinstance(remote_job_0, str) and len(remote_job_0) > 0

    # one job to get stuck
    system.run(['mkdir', '-p', system.remote_rundir + '/stage_dummy_queued_3'])

    queued_job_resources[1]['max_time'] = '5m'
    r = Resources(**(queued_job_resources[1]))
    partition, node_dict = r.find_nodes(system.partitions)

    remote_job_1 = sched.submit(id='dummy_id_3', remote_dir=system.remote_rundir + '/stage_dummy_queued_3', partition=partition,
                              commands=['pwd', 'echo BOB', 'sleep 30'], max_time=r.max_time,
                              header=system.queuing_sys_header, node_dict=node_dict)
    assert isinstance(remote_job_1, str) and len(remote_job_1) > 0

    # make sure it's queued
    status = sched.status(remote_job_1)
    assert status[remote_job_1] == 'queued'

    # hold and confirm it's held
    time.sleep(5)
    sched.hold(remote_job_1)

    status = sched.status(remote_job_1)
    assert status[remote_job_1] == 'held'

    # release and confirm it's back to queued
    time.sleep(5)
    sched.release(remote_job_1)

    status = sched.status(remote_job_1)
    assert status[remote_job_1] == 'queued'

    # cancel both and confirm they are done
    sched.cancel(remote_job_0)
    sched.cancel(remote_job_1)

    time.sleep(5)

    status = sched.status(remote_job_0)
    assert status[remote_job_0] == 'done'

    status = sched.status(remote_job_1)
    assert status[remote_job_1] == 'done'
