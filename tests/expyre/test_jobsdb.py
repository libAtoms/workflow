import sqlite3
from wfl.expyre.jobsdb import JobsDB

import pytest
pytestmark = pytest.mark.remote


def _clean(job):
    job_clean = job.copy()
    print('job_clean', job_clean)
    del job_clean['creation_time']
    del job_clean['status_time']
    return job_clean

def test_name(tmp_path):
    db = JobsDB(tmp_path / 'expyre.db')
    db.add('job1', 'task', 'rundir_1')
    db.add('job2', 'task', 'rundir_2')
    db.add('job3', 'othertask', 'rundir_3')

    assert len(list(db.jobs())) == 3
    assert len(list(db.jobs(name='task'))) == 2
    assert len(list(db.jobs(name='.*task'))) == 3
    assert len(list(db.jobs(name='other.*'))) == 1
    assert len(list(db.jobs(name='bob'))) == 0


def test_status(tmp_path):
    db = JobsDB(tmp_path / 'expyre.db')

    for stat in JobsDB.possible_status:
        db.add(f'job_{stat}', 'task', 'rundir_1', status=stat)

    for j in db.jobs():
        print('job', j)

    # one status
    assert set([j['status'] for j in db.jobs(status='created')]) == set(['created'])
    assert set([j['status'] for j in db.jobs(status=['created'])]) == set(['created'])

    # set of two statuses
    assert set([j['status'] for j in db.jobs(status=['created', 'processed'])]) == set(['created', 'processed'])

    # status_group
    assert set([j['status'] for j in db.jobs(status='ongoing')]) == set(['created', 'submitted', 'started'])
    assert set([j['status'] for j in db.jobs(status=['ongoing'])]) == set(['created', 'submitted', 'started'])

    # status_group or single status
    assert set([j['status'] for j in db.jobs(status=['ongoing', 'processed'])]) == set(['created', 'submitted', 'started', 'processed'])

    # status_group and id
    assert set([j['status'] for j in db.jobs(status='ongoing', id='job_created')]) == set(['created'])

    # status_group and non-matching id
    assert set([j['status'] for j in db.jobs(status='ongoing', id='job_processed')]) == set([])


def test_basic(tmp_path):
    # create DB
    db = JobsDB(tmp_path / 'expyre.db')

    # add a couple of jobs
    db.add('job1', 'task', 'rundir_1')
    db.add('job2', 'task', 'rundir_2')

    # check #
    jobs = list(db.jobs())
    assert len(jobs) == 2

    # check a job
    job_2 = list(db.jobs())[1]
    assert _clean(job_2) == {'id': 'job2', 'name': 'task', 'from_dir': 'rundir_2', 'status': 'created',
                     'remote_id': None, 'remote_status': None, 'system': None}

    # update two fields
    db.update('job2', status='succeeded', system='sys')
    job_2 = list(db.jobs())[1]
    assert _clean(job_2) == {'id': 'job2', 'name': 'task', 'from_dir': 'rundir_2', 'status': 'succeeded',
                     'remote_id': None, 'remote_status': None, 'system': 'sys'}

    # remove
    db.remove('job1')
    jobs = list(db.jobs())
    assert len(jobs) == 1

    # make sure bad field fails
    try:
        db.update('job2', badfield='done')
        succeeded = True
    except sqlite3.OperationalError:
        succeeded = False
    assert not succeeded

    # make sure bad status failed
    try:
        db.update('job2', status='other')
        succeeded = True
    except AssertionError:
        succeeded = False
    assert not succeeded

    # try to reopen
    db = JobsDB(tmp_path / 'expyre.db')
    jobs = list(db.jobs())
    assert len(jobs) == 1
    assert _clean(jobs[0]) == {'id': 'job2', 'name': 'task', 'from_dir': 'rundir_2', 'status': 'succeeded',
                       'remote_id': None, 'remote_status': None, 'system': 'sys'}

