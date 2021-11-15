import sys
import os
import time
import itertools

import shutil
import tempfile
import pickle
import hashlib
import base64

from pathlib import Path

from . import config
from .subprocess import subprocess_run
from .resources import Resources


class ExPyRe:
    """Interface for remotely running python functions.  It pickles the function and
    its arguments, stages files (pickles as well as additional requested input files)
    to a local directory, submits to remote System (which stages out inputs, submits
    on remote queuing system, and stages outputs back in), and then unpickles results
    to return function result.  It also creates/updates the JobsDB entry for the job
    as it does these steps.

    Possible status (see JobsDB): created, submitted, started, succeeded/failed, processed, cleaned
    """

    def __init__(self, name, *, input_files=[], env_vars=[], pre_run_commands=[], post_run_commands=[],
                 output_files=[], try_restart_from_prev=True, hash_ignore=[],
                 function=None, args=[], kwargs={},
                 _from_db_info=None):
        """Create Queued Remote Function object, pickles function and inputs, stores files
        in local stage dir, and adds to job database.
        Parameters
        ----------
        name: str
            name of job
        input_files: list(str | Path), optional
            input files that function will need.  Relative paths without '..' path components
            files will be copied to same path relative to remote rundir on remote machine.
            Absolute paths will result in files copied into top level remote rundir
        env_vars: list(str), optional
            list of env vars to set in remote queuing script
        pre_run_commands: list(str), optional
            list of commands, one per line, to run at start of remote queuing script, after commands
            in config.json that are used for system
        post_run_commands: list(str), optional
            list of commands, one per line, to run at end of remote queuing script, after actual task
        output_files: list(str), optional
            output files to be copied back after evaluation is done
        try_restart_from_prev: bool, default True
            try to restart from previous call, based on hash of function, arguments, and input files
        hash_ignore: list(int or str), optional
            args elements (int) or kwargs items (str) to ignore when making hash to determine if run is
            identical to some previous one
        function: Callable
            function to call
        args: list
            positional arguments to function
        kwargs: dict
            keyword arguments to function
        _from_db_info: dict, optional (intended for internal use)
            restart is from db, and dict contains special arguments: remote_id, system_name, status, stage_dir
        """
        if 'EXPYRE_TIMING_VERBOSE' in os.environ:
            sys.stderr.write(f'ExPyRe {name} constructor start {time.time()}\n')
        # arguments that are used when creating from a database entry
        if _from_db_info is not None:
            self.remote_id = _from_db_info['remote_id']
            self.system_name = _from_db_info['system_name']
            self.status = _from_db_info['status']
            self.stage_dir = Path(_from_db_info['stage_dir'])
            return

        assert function is not None

        self.remote_id = None
        self.system_name = None
        self.status = 'created'
        # self.stage_dir is set below

        kwargs = kwargs.copy()
        input_files = [Path(f) for f in input_files]
        env_vars = env_vars.copy()
        pre_run_commands = pre_run_commands.copy()
        post_run_commands = post_run_commands.copy()

        # check for valid input/output filenames
        for f in input_files:
            if not f.is_absolute() and any([p == '..' for p in f.parts]):
                raise ValueError(f'Input path with ".." in input file "{f}" not supported')
        for f in output_files:
            if f.startswith('/'):
                raise ValueError(f'Absolute output path "{f}" not supported')

        # pickle function and arguments
        pickled_func = pickle.dumps((function, args, kwargs))

        if 'EXPYRE_TIMING_VERBOSE' in os.environ:
            sys.stderr.write(f'ExPyRe {name} constructor starting hash {time.time()}\n')
        # hash for unique identifier
        h = hashlib.sha256()
        # hash on function
        h.update((function.__module__ + function.__name__).encode())
        # hash on args, possibly ignoring some
        assert all([isinstance(arg, (int, str)) for arg in hash_ignore])
        for arg_i, arg in enumerate(args):
            if arg_i not in hash_ignore:
                h.update(pickle.dumps(arg))
        for arg_key in sorted(list(kwargs)):
            if arg_key not in hash_ignore:
                h.update(pickle.dumps((arg_key, kwargs[arg_key])))
        # hash on input filenames and content
        for infile in input_files:
            subfiles = [infile]
            if infile.is_dir():
                subfiles += sorted(infile.rglob('*'))
            for subfile in subfiles:
                # filename
                h.update(str(subfile).encode())
                if subfile.is_file():
                    # file contents
                    with open(subfile, 'rb') as fin:
                        h.update(fin.read())
        # create deterministic unique identifier that can be part of filename
        arghash = base64.urlsafe_b64encode(h.digest()).decode()

        self.recreated = False
        # check if this task was successfully run before with matching id (name + arghash)
        if try_restart_from_prev:
            # NOTE: following loop will not match status == 'processed' because such jobs are
            # not guaranteed to have results available.  Is it a good idea to try to use those,
            # if results actually seem to be available?
            for job in config.db.jobs(status='can_produce_results', id=f'{name}_{arghash}_.*'):
                old_stage_dir = Path(job['from_dir'])
                # this also never happen
                if job['status'] == 'succeeded' and not (old_stage_dir / '_expyre_job_succeeded').exists():
                    raise RuntimeError(f'Found previously run job with matching id "{job["id"]}" '
                                       f'and status{job["status"]} but _succeeded file does not exist')

                # confirm that name is consistent, which it has to be since it's part of id
                assert name == job['name']

                # reconstruct job
                self.stage_dir = old_stage_dir

                # confirm that JobsDB id and self.id, as they must be since self.id
                # is derived from self.stage_dir (that's why self.stage_dir has to be set
                # first) but they are stored separately in JobsDB
                assert self.id == job['id']

                # save remaining attributes
                self.status = job['status']
                self.remote_id = job['remote_id']
                self.system_name = job['system']

                # set flag in case later routines need to treat it specially
                self.recreated = True
                return

        # didn't find an old run, need to create new and unique stage dir
        self.stage_dir = Path(tempfile.mkdtemp(prefix=f'run_{name}_{arghash}_', dir=config.root))
        # id is property derived from stage dir, so it is also unique

        if 'EXPYRE_TIMING_VERBOSE' in os.environ:
            sys.stderr.write(f'ExPyRe {name} constructor starting pickle {time.time()}\n')
        # write pickled function and arguments
        with open(self.stage_dir / '_expyre_task_in.pckl', 'wb') as fout:
            fout.write(pickled_func)

        if len(output_files) > 0:
            with open(self.stage_dir / '_expyre_output_files', 'w') as fout:
                fout.write('\n'.join(output_files) + '\n')

        if 'EXPYRE_TIMING_VERBOSE' in os.environ:
            sys.stderr.write(f'ExPyRe {name} constructor starting stage in files {time.time()}\n')
        # stage in input files
        # NOTE: does this require more thought?
        for f in input_files:
            if f.is_absolute():
                # copy absolute path into stage_dir stripping all leading components
                ExPyRe._copy(None, self.stage_dir, f)
            else:
                ExPyRe._copy(Path.cwd(), self.stage_dir, f)
        sys.stderr.write(f'ExPyRe {name} constructor done stage in files {time.time()}\n')


        # script commands to touch file indicating job has started
        pre_run_commands = ['touch _expyre_job_started', '('] + pre_run_commands

        # commands to set extra remote env vars
        for env_var in env_vars:
            if '=' in env_var:
                # form is already var=value
                pre_run_commands.append(f'export {env_var}')
            else:
                # get value from current environment
                pre_run_commands.append(f'export {env_var}={os.environ["env_var"]}')

        # save to file in stage dir
        with open(self.stage_dir / '_expyre_pre_run_commands', 'w') as fout:
            fout.write('\n'.join(pre_run_commands) + '\n')

        # Below always write to temporary file and then mv to try to make creation of final files more atomic

        # create core of remote job script
        with open(self.stage_dir / '_expyre_script_core.py', 'w') as fout:
            fout.write('try:\n'
                       '    import pickle, traceback, sys\n'
                       '    with open("_expyre_task_in.pckl", "rb") as fin:\n'
                       '        (function, args, kwargs) = pickle.load(fin)\n'
                       '    stdout_orig = sys.stdout\n'
                       '    stderr_orig = sys.stderr\n'
                       '    sys.stdout = open("_expyre_stdout", "w")\n'
                       '    sys.stderr = open("_expyre_stderr", "w")\n'
                       '    results = function(*args, **kwargs)\n'
                       '    sys.stdout = stdout_orig\n'
                       '    sys.stderr = stderr_orig\n'
                       '    with open(f"_tmp_expyre_job_succeeded", "wb") as fout:\n'
                       '        pickle.dump(results, fout)\n'
                       'except Exception as exc:\n'
                       '    with open(f"_expyre_error", "w") as fout:\n'
                       '        fout.write(f"{exc}\\n")\n'
                       '        traceback.print_exc(file=fout)\n'
                       '        raise\n')

        # commands to check status and create final _succeeded or _failed files
        post_run_commands = (['error_stat=$?'] + post_run_commands +
                             ['exit $error_stat', ')',
                              'error_stat=$?',
                              'if [ $error_stat == 0 ]; then',
                              '    if [ -e _tmp_expyre_job_succeeded ]; then',
                              '        mv _tmp_expyre_job_succeeded _expyre_job_succeeded',
                              '    else',
                              '        echo "No error code but _tmp_expyre_job_succeeded does not exist" > _tmp_expyre_job_failed',
                              '        if [ -f _expyre_error ]; then',
                              '            cat _expyre_error >> _tmp_expyre_job_failed',
                              '        fi',
                              '        mv _tmp_expyre_job_failed _expyre_job_failed',
                              '    fi',
                              'else',
                              '    if [ -e _expyre_error ]; then',
                              '        mv _expyre_error _expyre_job_failed',
                              '    else',
                              '        echo "UNKNOWN ERROR $error_stat" > _tmp_expyre_job_failed',
                              '        mv _tmp_expyre_job_failed _expyre_job_failed',
                              '    fi',
                              'fi',
                              ''])

        # save to file in stage dir
        with open(self.stage_dir / '_expyre_post_run_commands', 'w') as fout:
            fout.write('\n'.join(post_run_commands) + '\n')

        config.db.add(self.id, name=name, from_dir=str(self.stage_dir), status=self.status)

        if 'EXPYRE_TIMING_VERBOSE' in os.environ:
            sys.stderr.write(f'ExPyRe {name} constructor end {time.time()}\n')


    @staticmethod
    def _copy(in_dir, out_dir, file_glob):
        """ Copy files from in_dir to out_dir, including globs in filenames.
        If file_glob is absolute, file or directory is copied into out_dir with all of file's
            leading path components remove, i.e. file_glob -> out_dir / file_glob.name
        Otherwise, file_glob is copied into out_dir with all its (relative) components preserved,
            i.e. in_dir / file_glob -> out_dir / file_glob

        Parameters
        ----------
        in_dir: str or Path
            input directory, None if file_glob is absolute
        out_dir: str or Path
            output directory
        file_glob: str or Path
            glob of file or directory to copy (recursively), absolute iff in_dir is None

        """
        out_dir = Path(out_dir)
        file_glob = Path(file_glob)

        if file_glob.is_absolute():
            strip_leading = True
            assert in_dir is None

            # make in_dir root and file_glob relative to that, so that in_dir.glob(file_glob) works
            in_dir = Path(file_glob.root)
            file_glob = str(file_glob).replace(str(file_glob.root), '', 1)
        else:
            strip_leading = False
            assert in_dir is not None

            in_dir = Path(in_dir)
            assert in_dir.is_dir()

        in_files = list(in_dir.glob(str(file_glob)))
        if len(in_files) == 0:
            raise RuntimeError(f'File glob "{file_glob}" in input_files does not match any files')
        for in_file in in_files:
            if strip_leading:
                rel_out_file = in_file.name
            else:
                rel_out_file = in_file.relative_to(in_dir)

            out_file = out_dir / rel_out_file

            out_file.parent.mkdir(parents=True, exist_ok=True)
            try:
                shutil.copytree(in_file, out_file, dirs_exist_ok=True)
            except NotADirectoryError:
                shutil.copy(in_file, out_file)


    @property
    def id(self):
        return self.stage_dir.name.replace('run_', '', 1)


    @staticmethod
    def from_jobsdb(db_jobs):
        """Create a list of ExPyRe objects from JobsDB records
        Parameters:
        db_jobs: dict or list(dict)
            one or more jobs records returned from JobsDB.jobs()

        Returns
        -------
        list(ExPyRe): created objects
        """
        if isinstance(db_jobs, dict):
            db_jobs = [db_jobs]

        expyres = []
        for j in db_jobs:
            assert isinstance(j, dict)
            expyres.append(ExPyRe(name=j['name'],
                                    _from_db_info={'remote_id': j['remote_id'], 'system_name': j['system'],
                                                   'stage_dir': j['from_dir'], 'status': j['status']}))
        return expyres


    def start(self, resources, system_name=os.environ.get('EXPYRE_SYS', None),
              exact_fit=True, partial_node=False, python_cmd='python3'):
        """Start a job on a remote machine
        Parameters
        ----------
        resoures: dict or Resources
            resources to use for job, either Resources or dict of Resources constructor kwargs
        system_name: str
            name of system in config.systems
        exact_fit: bool, default True
            use only nodes that exactly match number of tasks
        partial_node: bool, default False
            allow jobs that take less than an entire node
        python_cmd: std, default python3
            name of python interpreter to use on remote machine
        """
        if self.status != 'created':
            assert self.recreated
            # Only newly created jobs can be started.  At any later state it just confuses things,
            # but this might be a call after constructor recreated job from old stage dir and database
            # record, and if so just return
            return

        if 'EXPYRE_TIMING_VERBOSE' in os.environ:
            sys.stderr.write(f'ExPyRe {self.id} start() start {time.time()}\n')

        if isinstance(resources, dict):
            resources = Resources(**resources)

        self.system_name = system_name
        system = config.systems[self.system_name]
        with open(self.stage_dir / '_expyre_pre_run_commands') as fin:
            pre_run_commands = fin.readlines()
        with open(self.stage_dir / '_expyre_post_run_commands') as fin:
            post_run_commands = fin.readlines()
        if 'EXPYRE_TIMING_VERBOSE' in os.environ:
            sys.stderr.write(f'ExPyRe {self.id} start() calling system.submit {time.time()}\n')
        self.remote_id = system.submit(self.id, self.stage_dir, resources=resources,
                                       commands=(pre_run_commands + [f'{python_cmd} _expyre_script_core.py'] +
                                                 post_run_commands),
                                       exact_fit=exact_fit, partial_node=partial_node)
        if 'EXPYRE_TIMING_VERBOSE' in os.environ:
            sys.stderr.write(f'ExPyRe {self.id} start() done system.submit {time.time()}\n')

        self.status = 'submitted'
        config.db.update(self.id, status=self.status, system=self.system_name, remote_id=self.remote_id)

        if 'EXPYRE_TIMING_VERBOSE' in os.environ:
            sys.stderr.write(f'ExPyRe {self.id} start() end {time.time()}\n')


    def sync_results(self, sync_all=True, force_sync=False, verbose=False):
        """Sync files associated with results from remote machine to local stage dirs
        Parameters
        ----------
        sync_all: bool, default True
            sync files for all jobs on same system, to minimize number of calls to remote copy
        force_sync: bool, default False
            sync files even if this job's status indicates that this has already been done
        verbose: bool, default False
            verbose output
        """
        # sync all jobs
        if sync_all:
            job_id = None
        else:
            job_id = self.id

        # sync even if already done
        if force_sync:
            status = None
        else:
            status = 'ongoing'

        jobs_to_sync = list(config.db.jobs(system=self.system_name, id=job_id, status=status))

        def _grouper(n, iterable):
            it = iter(iterable)
            while True:
                chunk = tuple(itertools.islice(it, n))
                if not chunk:
                    return
                yield chunk

        if len(jobs_to_sync) > 0:
            system = config.systems[self.system_name]
            # get remote files
            for job_group in _grouper(250, jobs_to_sync):
                system.get_remotes(self.stage_dir.parent, subdir_glob=[Path(j['from_dir']).name for j in job_group],
                                   verbose=verbose)

            # get remote statuses and update in JobsDB
            status_of_remote_id = system.scheduler.status([j['remote_id'] for j in jobs_to_sync], verbose=verbose)
            for j in jobs_to_sync:
                config.db.update(j['id'], remote_status=status_of_remote_id[j['remote_id']])


    def clean(self, wipe=False, dry_run=False, verbose=False):
        """clean the local and remote stage directories
        Parameters
        ----------
        wipe: bool, default False
            wipe directory completely, opposed to just writing CLEANED into files that could be large
            like python function input and output (NOTE: other staged in files or files that are created
            are not cleaned if wipe=False)
        dry_run: bool, default False
            dry run only, print what will happen but do not actually delete or overwrite anything
        verbose: bool, default False
            verbose output
        """

        if self.system_name is not None:
            system = config.systems[self.system_name]
        else:
            system = None
        if wipe:
            if system is not None:
                # delete remote stage dir
                system.clean_rundir(self.stage_dir, None, dry_run=dry_run, verbose=verbose or dry_run)
            # delete local stage dir
            subprocess_run(None, ['rm', '-rf', str(self.stage_dir)], dry_run=dry_run, verbose=verbose or dry_run)
        else:
            if system is not None:
                # clean remote stage dir
                system.clean_rundir(self.stage_dir, ['_expyre_task_in.pckl', '_expyre_job_succeeded'],
                                dry_run=dry_run, verbose=verbose or dry_run)
            if dry_run:
                print(f"dry-run overwrite local dirs {self.stage_dir / '_expyre_task_in.pckl'} and "
                      f"{self.stage_dir / '_expyre_job_succeeded'}, and create "
                      f"{self.stage_dir / '_expyre_job_cleaned'}")
            else:
                # clean local stage dir
                with open(self.stage_dir / '_expyre_task_in.pckl', 'w') as fout:
                    fout.write('CLEANED\n')
                f = self.stage_dir / '_expyre_job_succeeded'
                if f.exists():
                    with open(f, 'w') as fout:
                        fout.write('CLEANED\n')
                with open(self.stage_dir / '_expyre_job_cleaned', 'w') as fout:
                    fout.write('CLEANED\n')

        if not dry_run:
            self.status = 'cleaned'
            config.db.update(self.id, status=self.status)


    def get_results(self, timeout=3600, check_interval=30, sync=True, sync_all=True, force_sync=False, quiet=False, verbose=False):
        """Get results from a remote job
        Parameters
        ----------
        timeout: int, default 3600
            Max time to wait for job to complete, in sec
        check_interval: int, default 30
            Time to wait between checks of job completion
        sync: bool, default True
            Synchronize remote files before checking for results
            Note that if this is False and job is finished on remote system but output files haven't been
                previously synchronize, this will wait one check_interval then raise an error
        sync_all: bool, default True
            Sync files from all jobs (not just this one), to reduce number of separate remote copy calls
        force_sync: bool, default False
            Sync remote files even if job's DB status indicates that this was already done.
            Note: together with sync_all this can lead to rsync having to compare many files.
        quiet: bool, default False
            No progress info
        verbose: bool, default False
            Verbose output (from remote system/scheduler commands)

        Returns
        -------
        return, stdout, stderr:
            value of function
            string containing stdout during function
            string containing stderr during function
        """
        if self.status == 'processed':
            raise RuntimeError(f'Job {self.id} has been processed, results are no longer available')

        system = config.systems[self.system_name]
        start_time = time.time()
        problem_last_chance = False
        out_of_time = False
        n_iter = 0
        # this is a messy state machine - only fairly sure that there are no deadlocks or infinite loops
        while True:
            # this test (i.e. manually looking for _expyre_job_[succeeded, failed]) seems a bit messy
            if (sync and self.status != 'succeeded' and self.status != 'failed' and
                not (self.stage_dir / '_expyre_job_succeeded').exists() and
                not (self.stage_dir / '_expyre_job_failed').exists()):

                # if not quiet:
                    # if n_iter > 0:
                        # sys.stderr.write('\n')
                    # sys.stderr.write(f'Doing sync when getting job {self.id}\n')
                self.sync_results(sync_all, force_sync, verbose=verbose)

            # get remote status of job, was set by call to self.sync_results() just above
            # remote_status: queued, held,       running,    done, failed, timeout, other
            remote_status = list(config.db.jobs(id=self.id))[0]['remote_status']

            # update state depending on presence of various progress files and remote status
            if (self.stage_dir / '_expyre_job_succeeded').exists():
                # job created final succeeded file
                assert remote_status not in ['queued', 'held']
                try:
                    with open(self.stage_dir / '_expyre_job_succeeded', 'rb') as fin:
                        results = pickle.load(fin)
                    with open(self.stage_dir / '_expyre_stdout') as fin:
                        stdout = fin.read()
                    with open(self.stage_dir / '_expyre_stderr') as fin:
                        stderr = fin.read()
                except Exception as exc:
                    raise RuntimeError(f'Job {self.id} got "_succeeded" file, but failed to parse it with error {exc}')
                self.status = 'succeeded'
            elif (self.stage_dir / '_expyre_job_failed').exists():
                # job created final failed file
                assert remote_status not in ['queued', 'held']
                with open(self.stage_dir / '_expyre_job_failed') as fin:
                    error_msg = ''.join(fin.readlines())
                self.status = 'failed'
            else:
                if (self.stage_dir / '_expyre_job_started').exists():
                    self.status = 'started'
                # job does not _appear_ to have finished
                if remote_status not in ['queued', 'held', 'running']:
                    # problem - job does not seem to be queued (even held) or running
                    if problem_last_chance:
                        # already on last chance, giving up
                        # NOTE: should this really be 'failed', or some other status that the calling
                        # program can deal with separately?
                        self.status = 'failed'
                        config.db.update(self.id, status=self.status)
                        raise RuntimeError(f'Job {self.id} got remote status {remote_status} which is not '
                                            '"queued", "running", or "held", but neither "_succeeded" nor '
                                            '"_failed" file exists')
                    # give it one more chance, perhaps queuing system status and file are slow to sync to head node
                    problem_last_chance = True
                else:
                    # No apparent problem, just not done yet, leave status as is, but check for timeout
                    if out_of_time:
                        if not quiet and n_iter > 0:
                            sys.stderr.write('\n')
                            sys.stderr.flush()
                        raise TimeoutError

            # update status in database
            config.db.update(self.id, status=self.status)

            # return if succeeded or failed
            if self.status == 'succeeded':
                # stage out remotely created files
                # should we do this for failed calls?
                if (self.stage_dir / '_expyre_output_files').exists():
                    with open(self.stage_dir / '_expyre_output_files') as fin:
                        for in_file in [f.replace('\n', '') for f in fin.readlines()]:
                            ExPyRe._copy(self.stage_dir, Path.cwd(), in_file)

                if not quiet and n_iter > 0:
                    # newline after one or more 'q|r' progress characters
                    sys.stderr.write('\n')
                    sys.stderr.flush()
                return results, stdout, stderr
            elif self.status == 'failed':
                if not quiet and n_iter > 0:
                    # newline after one or more 'q|r' progress characters
                    sys.stderr.write('\n')
                    sys.stderr.flush()
                # should (can?) we pickle the exception and reraise it here?
                raise RuntimeError(f'Remote job {self.id} failed with status {remote_status} error_msg {error_msg}')

            out_of_time = time.time() - start_time > timeout

            if not quiet:
                if n_iter == 0:
                    sys.stderr.write(f'Waiting for job {self.id}: \n')
                    sys.stderr.flush()

                # progress info to stderr
                if n_iter % 10 == 10 - 1:
                    sys.stderr.write(f'{(n_iter // 10) % 10}')
                else:
                    sys.stderr.write(remote_status[0])
                if n_iter % 100 == 100 - 1:
                    sys.stderr.write('\n')
                sys.stderr.flush()

            # wait for next check
            time.sleep(check_interval)
            n_iter += 1


    def mark_processed(self):
        """Mark job as processed (usually after results have been stored someplace)
        """
        self.status = 'processed'
        config.db.update(self.id, status=self.status)


    def __str__(self):
        return f'{self.id} system={self.system_name} remote_id={self.remote_id} status={self.status} stage_dir={self.stage_dir}'
