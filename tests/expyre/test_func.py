import sys
import os
import re
import subprocess
import time

import pytest
pytestmark = pytest.mark.remote

from pathlib import Path

from wfl.expyre.resources import Resources


def test__copy_stage_in(tmp_path, monkeypatch):
    from wfl.expyre.func import ExPyRe

    # make initial dir, and its subdirectories with files
    (tmp_path / 'init_dir').mkdir()
    monkeypatch.chdir(tmp_path / 'init_dir')

    for sd in ['subdir1', 'subdir2']:
        (tmp_path / 'init_dir' / sd).mkdir()
        for sf in ['file1', 'file2']:
            with open(tmp_path / 'init_dir' / sd / sf, 'w') as fout:
                fout.write('test\n')


    # copy absolute path file to stage dir
    (tmp_path / 'job_stage_dir_1').mkdir()
    ExPyRe._copy(None, tmp_path / 'job_stage_dir_1', tmp_path / 'init_dir' / 'subdir1' / 'file1')
    assert [str(f) for f in (tmp_path / 'job_stage_dir_1').rglob('*')] == [ str(tmp_path / 'job_stage_dir_1' / 'file1') ]

    # copy absolute path dir to stage dir
    (tmp_path / 'job_stage_dir_2').mkdir()
    ExPyRe._copy(None, tmp_path / 'job_stage_dir_2', tmp_path / 'init_dir' / 'subdir1')
    assert set([str(f) for f in (tmp_path / 'job_stage_dir_2').rglob('*')]) == set([ str(tmp_path / 'job_stage_dir_2' / 'subdir1'),
                                                                                     str(tmp_path / 'job_stage_dir_2' / 'subdir1' / 'file1'),
                                                                                     str(tmp_path / 'job_stage_dir_2' / 'subdir1' / 'file2') ])

    # copy rel path file to stage dir
    (tmp_path / 'job_stage_dir_3').mkdir()
    ExPyRe._copy(Path.cwd(), tmp_path / 'job_stage_dir_3', 'subdir1/file1')
    assert set([str(f) for f in (tmp_path / 'job_stage_dir_3').rglob('*')]) == set([ str(tmp_path / 'job_stage_dir_3' / 'subdir1'),
                                                                                     str(tmp_path / 'job_stage_dir_3' / 'subdir1' / 'file1') ])

    # copy rel path dir to stage dir
    (tmp_path / 'job_stage_dir_4').mkdir()
    ExPyRe._copy(Path.cwd(), tmp_path / 'job_stage_dir_4', 'subdir1')
    assert set([str(f) for f in (tmp_path / 'job_stage_dir_4').rglob('*')]) == set([ str(tmp_path / 'job_stage_dir_4' / 'subdir1'),
                                                                                     str(tmp_path / 'job_stage_dir_4' / 'subdir1' / 'file1'),
                                                                                     str(tmp_path / 'job_stage_dir_4' / 'subdir1' / 'file2') ])

    # copy rel path glob
    (tmp_path / 'job_stage_dir_5').mkdir()
    ExPyRe._copy(Path.cwd(), tmp_path / 'job_stage_dir_5', 'subdir*/file1')
    assert set([str(f) for f in (tmp_path / 'job_stage_dir_5').rglob('*')]) == set([ str(tmp_path / 'job_stage_dir_5' / 'subdir1'),
                                                                                     str(tmp_path / 'job_stage_dir_5' / 'subdir1' / 'file1'),
                                                                                     str(tmp_path / 'job_stage_dir_5' / 'subdir2'),
                                                                                     str(tmp_path / 'job_stage_dir_5' / 'subdir2' / 'file1') ])


def test__copy_stage_out(tmp_path, monkeypatch):
    from wfl.expyre.func import ExPyRe

    # make initial dir, and its subdirectories with files
    (tmp_path / 'init_dir').mkdir()
    monkeypatch.chdir(tmp_path / 'init_dir')

    (tmp_path / 'job_stage_dir').mkdir()
    for sd in ['subdir1', 'subdir2']:
        (tmp_path / 'job_stage_dir' / sd).mkdir()
        for sf in ['file1', 'file2']:
            with open(tmp_path / 'job_stage_dir' / sd / sf, 'w') as fout:
                fout.write('test\n')

    # copy from absolute stage_dir
    ExPyRe._copy(tmp_path.resolve() / 'job_stage_dir', Path.cwd(), 'subdir1/file1')
    assert set([str(f) for f in Path.cwd().rglob('*')]) == set([ str(Path.cwd() / 'subdir1'),
                                                                 str(Path.cwd() / 'subdir1' / 'file1') ])


def test_work(expyre_config):
    from wfl.expyre import config

    for sys_name in config.systems:
        if sys_name.startswith('_'):
            continue

        sys.stderr.write(f'Test working job {sys_name}\n');

        do_work(sys_name)


def test_clean(expyre_config, cleanremote):
    from wfl.expyre import config

    dry_run = not cleanremote

    for sys_name in config.systems:
        if sys_name.startswith('_'):
            continue

        sys.stderr.write(f'Test job {sys_name} cleanup\n');

        do_clean(sys_name, dry_run)

    if dry_run:
        print('\n')
        pytest.xfail('clean test dry run only')


def test_restart(expyre_config):
    from wfl.expyre import config

    for sys_name in config.systems:
        if sys_name.startswith('_'):
            continue

        sys.stderr.write(f'Test restart job {sys_name}\n');

        do_restart(sys_name)


def test_stdouterr(expyre_config):
    from wfl.expyre import config

    for sys_name in config.systems:
        if sys_name.startswith('_'):
            continue

        sys.stderr.write(f'Test restart job {sys_name}\n');

        do_stdouterr(sys_name)


def do_stdouterr(sys_name):
    from wfl.expyre.config import root, db

    from wfl.expyre.func import ExPyRe

    xpr = ExPyRe('test_stdout', function=print, args=['stdout content'])

    xpr.start(resources=Resources(n=(1, 'nodes'), max_time='5m'), system_name=sys_name)
    results, stdout, stderr = xpr.get_results(check_interval=10)

    assert results is None
    assert stdout == 'stdout content\n'
    assert stderr == ''


    import warnings
    xpr = ExPyRe('test_stderr', function=warnings.warn, args=['stderr warning'])

    xpr.start(resources=Resources(n=(1, 'nodes'), max_time='5m'), system_name=sys_name)
    results, stdout, stderr = xpr.get_results(check_interval=10)

    assert results is None
    assert stdout == ''
    assert 'UserWarning: stderr warning' in stderr


def do_work(sys_name):
    from wfl.expyre.config import root, db

    # must do this here rather than outside of functions because wfl.expyre.func imports config, and it's not
    # yet set up by conftest.py if imported outside the test function
    from wfl.expyre.func import ExPyRe

    xpr = ExPyRe('test', function=sum, args=[[1, 2, 3]])

    print('job id', xpr.id)

    assert (Path(root) / f'run_{xpr.id}').exists()
    assert (Path(root) / f'run_{xpr.id}' / '_expyre_script_core.py').exists()
    assert (Path(root) / f'run_{xpr.id}' / '_expyre_task_in.pckl').exists()

    xpr.start(resources=Resources(n=(1, 'nodes'), max_time='5m'), system_name=sys_name)

    # wipe existing job objects
    xpr = None
    # recreate from JobsDB
    xpr = ExPyRe.from_jobsdb(db.jobs(status=['ongoing', 'unprocessed']))
    assert len(xpr) == 1
    xpr = xpr[0]

    r, stdout, stderr = xpr.get_results(check_interval=10)
    xpr.mark_processed()

    assert r == sum([1,2,3])


def do_clean(sys_name, dry_run):
    from wfl.expyre.config import systems
    from wfl.expyre.func import ExPyRe

    if dry_run:
        print('do_clean', sys_name)

    system = systems[sys_name]

    xpr = ExPyRe('test', function=sum, args=[list(range(1000))])
    xpr.start(resources=Resources(n=(1, 'nodes'), max_time='5m'), system_name=sys_name)
    r, stdout, stderr = xpr.get_results(check_interval=10)

    if dry_run:
        print('do_clean no wipe')
    xpr.clean(dry_run=dry_run)

    if not dry_run:
        local_post_clean = ''
        for f in xpr.stage_dir.glob('*'):
            p = subprocess.run(['wc', '-c', f'{f}'], capture_output=True)
            local_post_clean += p.stdout.decode()
        stdout, _ = system.run(args=['bash'], script=f'cd {system.remote_rundir}/{xpr.stage_dir.name} && wc -c *\n')
        remote_post_clean = stdout

        # check for files that have been CLEANED
        for l in local_post_clean.splitlines() + remote_post_clean.splitlines():
            if re.search('_expyre_job-test_.*-succeeded', l) or re.search('_expyre_task_in.pckl', l):
                s = int(l.strip().split()[0])
                assert dry_run or s == len('CLEANED\n')

    if dry_run:
        print('do_clean with wipe')
    xpr.clean(wipe=True, dry_run=dry_run)

    # check that stage directories are gone
    assert dry_run or not xpr.stage_dir.exists()
    if not dry_run:
        stdout, _ = system.run(args=['bash'],
                               script=f'if [ -f {system.remote_rundir}/{xpr.stage_dir.name} ]; then\n'
                                       '    echo yes\n'
                                       'else\n'
                                       '    echo no\n'
                                       'fi\n')
        assert stdout.strip() == 'no'


def do_restart(sys_name):
    from wfl.expyre.config import db

    # must do this here rather than outside of functions because wfl.expyre.func imports config, and it's not
    # yet set up by conftest.py if imported outside the test function
    from wfl.expyre.func import ExPyRe

    xpr = ExPyRe('rerun', function=time.sleep, args=[30])
    print('initial job id', xpr.id)

    t0 = time.time()
    xpr.start(resources=Resources(n=(1, 'nodes'), max_time='5m'), system_name=sys_name)
    res, stdout, stderr = xpr.get_results()
    print('status after get_results', xpr.status)

    assert time.time() - t0 > 29

    # wipe existing job objects
    xpr = None

    # auto recreate
    xpr = ExPyRe('rerun', function=time.sleep, args=[30])
    print('restarted job id', xpr.id)
    print('status after recreation', xpr.status)

    t0 = time.time()
    xpr.start(resources=Resources(n=(1, 'nodes'), max_time='5m'), system_name=sys_name)
    res, stdout, stderr = xpr.get_results()

    # recreated run should be nearly instantaneous
    assert time.time() - t0 < 15

    # mark as processed, so next call to this function won't reuse results
    xpr.mark_processed()
