import time
import sqlite3
import re
from pathlib import Path


class _SQLite3Val():
    def __init__(self, v):
        self.v = v

    def __str__(self):
        if self.v is None:
            return 'NULL'
        elif isinstance(self.v, (int, float)):
            return f"{self.v}"
        else:
            return f"'{self.v}'"


class JobsDB:
    """Database of jobs, currently implemented with sqlite.  Saves essential information
    on jobs, including local and remote id, local directory it's staged to, system
    its been submitted to, and status.

    Status: created - job has been created, but not yet submitted to run
            submitted - job has been submitted to some queuing system
            started - job has started running
            succeeded - job has finished successfully, and results are available
            failed - job has failed
            processed - job has been processed, and results no longer need to be saved
    """

    possible_status = ['created', 'submitted', 'started', 'succeeded', 'failed', 'processed', 'cleaned']
    status_group = {'ongoing': ['created', 'submitted', 'started'],
                    'unprocessed': ['succeeded', 'failed'],
                    'can_produce_results': ['created', 'submitted', 'started', 'succeeded']}


    def _execute(self, cmd, retry_n=3, retry_wait=1):
        for i in range(retry_n):
            try:
                with self.db:
                    res = self.db.execute(cmd)
                return res
            except sqlite3.OperationalError as exc:
                if 'database is locked' in str(exc):
                    exc_str = f'{type(exc)} {exc}'
                    time.sleep(retry_wait)
                else:
                    raise
            except Exception as exc:
                raise

        raise RuntimeError(f'Repeatedly got {exc_str} from cmd {cmd}')


    def __init__(self, db_filename):
        """Create JobsDB obect
        Parameters
        ----------
        db_filename: str
            database file
        """
        self.db_filename = db_filename

        self.columns =      ['id',   'name', 'from_dir', 'status', 'system', 'remote_id', 'remote_status', 'creation_time', 'status_time']
        self.column_types = ['TEXT', 'TEXT', 'TEXT',     'TEXT',   'TEXT',   'TEXT',      'TEXT',          'DATE',          'DATE']
        for col_i, col in enumerate(self.columns):
            setattr(self, col+'_col', col_i)

        if Path(self.db_filename).exists():
            # just connect to existing database
            self.db = sqlite3.connect(db_filename)
            # make sure database can be minimally accessed
            try:
                _ = self._execute("SELECT * FROM jobs")
            except:
                raise RuntimeError(f"Failed to read list of jobs from existing JobsDB file {self.db_filename}")
        else:
            # connect should create the file here
            self.db = sqlite3.connect(db_filename)
            # create actual database table
            self._execute("CREATE TABLE jobs (" + ', '.join([c + ' ' + t for c, t in zip(self.columns, self.column_types)])+")")


    def add(self, id, name, from_dir, status='created', system=None, remote_id=None, remote_status=None):
        """Add a job to the DB
        Parameters
        ----------
        id: str
            unique id for job (fails if id already exists)
        name: str
            name for job
        from_dir: str/Path
            path to directory job is to run from
        status: str, default 'created'
            status of job
        system: str, optional
            system job is running on
        remote_id: str, optional
            remote id on system
        remote_status: str, optional
            remote status on system
        """
        assert status in JobsDB.possible_status

        rows = list(self._execute(f"SELECT * FROM jobs WHERE id = '{id}'"))
        if len(rows) != 0:
            raise ValueError(f"JobsDB trying to add job {id} which already exists")

        self._execute(f'INSERT into jobs(id, name, from_dir, status, system, remote_id, remote_status, creation_time, status_time) '
                      f'values ({_SQLite3Val(id)}, {_SQLite3Val(name)}, {_SQLite3Val(from_dir)}, '
                      f'{_SQLite3Val(status)}, {_SQLite3Val(system)}, {_SQLite3Val(remote_id)}, {_SQLite3Val(remote_status)}, '
                      f'{_SQLite3Val(int(time.time()))}, {_SQLite3Val(None)})')


    def remove(self, id):
        """Remove a job from the DB
        Parameters
        ----------
        id: str
            unique id of job to remove
        """
        rows = list(self._execute(f"SELECT * FROM jobs WHERE id = '{id}'"))
        if len(rows) != 1:
            raise ValueError(f"JobsDB trying to remove job {id}, found {len(rows)} such entries")

        self._execute(f"DELETE FROM jobs WHERE id = '{id}'")


    def update(self, id, /, **kwargs):
        """Update some field of job
        Parameters
        ----------
        id: str
            unique id of job to update
        from_dir, status, system, remote_id, remote_status: str
            field(s) to update
        """
        if 'status' in kwargs:
            assert kwargs['status'] in JobsDB.possible_status

        rows = list(self._execute(f"SELECT * FROM jobs WHERE id = '{id}'"))
        if len(rows) != 1:
            raise ValueError(f"JobsDB trying to update job {id}, found {len(rows)} such entries")

        if 'status' in kwargs:
            kwargs['status_time'] = int(time.time())

        self._execute("UPDATE jobs SET " +
                      ", ".join([f"{k}={_SQLite3Val(v)}" for k, v in kwargs.items()]) +
                      f" WHERE id = '{id}'")


    def jobs(self, status=None, id=None, name=None, system=None, readable=True):
        """Iterate through jobs
        Parameters
        ----------
        status: str or list(str), default None
            if not None, only report on jobs that match any status
        id: str or list(str), default None
            if present, include only jobs with id that match regexps in this list
        name: str or list(str), default None
            if present, include only jobs with name that matches regexps in this list
        system: str or list(str), default None
            if present, include only jobs with system in this list

        Returns
        -------
        Iterator of dicts with fields for all DB columns for each job that matches selection criteria.
        """
        if isinstance(status, str):
            status = [status]
        if isinstance(id, str):
            id = [id]
        if isinstance(name, str):
            name = [name]
        if isinstance(system, str):
            system = [system]

        if status is not None:
            assert all([stat in JobsDB.possible_status or stat in JobsDB.status_group for stat in status])

        def _col_match(col_res, col_val):
            return col_res is None or any([col_val is not None and re.search('^' + col_re + '$', col_val) for col_re in col_res])

        # do selection in python, not SQL query
        for row in self._execute('SELECT * FROM jobs'):
            if (_col_match(id, row[self.id_col]) and
                _col_match(system, row[self.system_col]) and
                _col_match(name, row[self.name_col]) and
                (status is None or
                 row[self.status_col] in status or
                 any([row[self.status_col] in JobsDB.status_group.get(stat_grp, []) for stat_grp in status]))):
                row = {k: v for k, v in zip(self.columns, row)}
                if readable:
                    if row['creation_time'] is not None:
                        row['creation_time'] = time.strftime('%Y-%m-%d %X', time.localtime(row['creation_time']))
                    if row['status_time'] is not None:
                        row['status_time'] = time.strftime('%Y-%m-%d %X', time.localtime(row['status_time']))
                yield row


    def __str__(self):
        s = f'JobsDB {self.db_filename}\n' + ' '.join(self.columns)
        jobs = list(self.jobs())
        if len(jobs) > 0:
            s += ('\n--------------------\n' +
                  '\n'.join([str(j) for j in jobs]))
        s += '\n--------------------'
        return s


    def unlock(self):
        # create tmp
        tmp_db_file = self.db_filename.parent / (self.db_filename.name + '.tmp')
        new_db = sqlite3.connect(tmp_db_file)

        # do backup
        with self.db:
            self.db.backup(new_db)

        # rename
        self.db.close()
        self.db_filename.rename(self.db_filename.parent / (self.db_filename.name + '.old'))
        tmp_db_file.rename(self.db_filename)

        # reinit saved pointers
        self.db = sqlite3.connect(self.db_filename)
