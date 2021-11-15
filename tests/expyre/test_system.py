import sys
import time

import pytest
pytestmark = pytest.mark.remote

from pathlib import Path

from wfl.expyre.resources import Resources

def test_system_same_machine(tmp_path, expyre_config):
    from wfl.expyre.config import systems

    for sys_name, system in systems.items():
        if sys_name.startswith('_'):
            continue

        sys.stderr.write(f'Testing system {sys_name}\n')

        do_system(tmp_path, system, 'dummy_job_'+sys_name)


def do_system(tmp_path, system, job_name):
    stage_dir = tmp_path / ('stage_' + job_name)
    stage_dir.mkdir()

    assert not (stage_dir / 'out').exists()

    remote_id = system.submit(job_name, stage_dir,
                           resources=Resources(n=(2, 'nodes'), max_time='5m'),
                           commands=['pwd', 'echo BOB > out', 'sleep 20'])

    # wait to finish
    print('remote_id', remote_id)
    status = system.scheduler.status(remote_id)
    while status[remote_id] != 'done':
        time.sleep(5)
        status = system.scheduler.status(remote_id)
        sys.stderr.write(f'status {status}\n')

    # make sure it's not failed
    assert status[remote_id] == 'done'

    system.get_remotes(tmp_path)

    # make sure output file got staged back in, and has correct content
    assert (stage_dir / 'out').exists()

    with open(stage_dir / 'out') as fin:
        lines = fin.readlines()
    assert ['BOB'] == [l.strip() for l in lines]

