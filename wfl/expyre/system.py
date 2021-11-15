import sys
import os
from pathlib import Path
import time

from .subprocess import subprocess_run, subprocess_copy
from .schedulers import schedulers
from . import util


class System:
    """Interface for a System that can run jobs remotely, including staging files from
    a local directory to a (config-specified) remote directory, submitting it with the correct
    kind of Scheduler, and staging back the results from the remote directory.  Does not report
    any other status directly.
    """
    def __init__(self, host, partitions, scheduler, header=[], no_default_header=False, commands=[],
                 rundir='run_expyre', rundir_extra=None, remsh_cmd=None):
        """Create a remote system object

        Parameters
        ----------
        host: str
            [username@]machine.fqdn
        partitions: dict
            dictionary describing partitions types
        scheduler: str, Scheduler
            type of scheduler
        header: list(str), optional
            list of batch system header to use in every job, typically for system-specific things
            like selecting nodes
        no_default_header: bool, default False
            do not automatically add default header fields, namely job name, partition/queue,
            max runtime, and stdout/stderr files
        commands: list(str), optional
            list of commands to run at start of every job on machine
        rundir: str, default 'run_expyre'
            path on remote machine to run in.  If absolute, used as if, and if relative, relative
            to (remote) home directory
        remsh_cmd: str, default EXPYRE_RSH or 'ssh'
            remote shell command to use with this system
        rundir_extra: str, default None
            extra string to add to remote_rundir, e.g. per-project part of path
        """
        self.host = host
        self.remote_rundir = rundir
        while self.remote_rundir.endswith('/'):
            self.remote_rundir = self.remote_rundir[:-1]
        if rundir_extra is not None:
            self.remote_rundir += '/' + rundir_extra
        self.partitions = partitions.copy() if partitions is not None else partitions
        self.queuing_sys_header = header.copy()
        self.no_default_header = no_default_header
        self.commands = commands.copy()
        self.remsh_cmd = util.remsh_cmd(remsh_cmd)
        self.initialized = False

        if isinstance(scheduler, str):
            self.scheduler = schedulers[scheduler](host, self.remsh_cmd)
        else:
            self.scheduler = scheduler(host)


    def run(self, args, script=None, shell='bash -lc', retry=None, in_dir='_HOME_', dry_run=False, verbose=False):
        # like subprocess_run, but filling in host and remsh command from self
        return subprocess_run(self.host, args, script=script, shell=shell, remsh_cmd=self.remsh_cmd,
                              retry=retry, in_dir=in_dir, dry_run=dry_run, verbose=verbose)


    def initialize_remote_rundir(self, verbose=False):
        if self.initialized:
            return

        self.run(['mkdir', '-p', str(self.remote_rundir)], verbose=verbose)
        self.initialized = True


    def _job_remote_rundir(self, stage_dir):
        return f'{self.remote_rundir}/{stage_dir.name}'


    def submit(self, id, stage_dir, resources, commands, exact_fit=True, partial_node=False, verbose=False):
        """Submit a job on a remote machine, including staging out files

        Parameters
        ----------
        id: str
            unique id for job
        stage_dir: str, Path
            directory in which files have been prepared
        resoures: Resources
            resources to use for job
        commands: list(str)
            commands to run in job script after per-machine commands
        exact_fit: bool, default True
            only match partitions that have nodes with exact match to number of tasks
        partial_node: bool, default False
            allow jobs that take less than an entire node

        Returns
        -------
        id of job on remote machine
        """
        if 'EXPYRE_TIMING_VERBOSE' in os.environ:
            sys.stderr.write(f'system {self.id} submit start {time.time()}\n')
        self.initialize_remote_rundir()

        partition, node_dict = resources.find_nodes(self.partitions, exact_fit=exact_fit,
                                                    partial_node=partial_node)
        commands = self.commands + commands

        stage_dir = Path(stage_dir)
        job_remote_rundir = self._job_remote_rundir(stage_dir)

        # make remote rundir, but fail if job-specific remote dir already exists
        self.run(['bash'],
                 script=f'if [ ! -d "{self.remote_rundir}" ]; then '
                        f'    echo "remote rundir \'{self.remote_rundir}\' does not exist" 1>&2; '
                         '    exit 1; '
                        f'elif [ -e "{job_remote_rundir}" ]; then '
                        f'    echo "remote job rundir \'{job_remote_rundir}\' already exists" 1>&2; '
                         '    exit 2; '
                         'else '
                        f'    mkdir -p "{job_remote_rundir}"; '
                         'fi', verbose=verbose)

        # stage out files
        # strip out final / from source path so that rsync creates stage_dir.name remotely under self.remote_rundir
        stage_dir_src = str(stage_dir)
        while stage_dir_src.endswith('/'):
            stage_dir_src = stage_dir_src[:-1]
        if 'EXPYRE_TIMING_VERBOSE' in os.environ:
            sys.stderr.write(f'system {self.id} submit start stage in {time.time()}\n')
        subprocess_copy(stage_dir_src, self.remote_rundir, to_host=self.host,
                        remsh_cmd=self.remsh_cmd, verbose=verbose)

        # submit job
        if 'EXPYRE_TIMING_VERBOSE' in os.environ:
            sys.stderr.write(f'system {self.id} submit start scheduler submit {time.time()}\n')
        try:
            r = self.scheduler.submit(id, str(job_remote_rundir), partition,
                                      commands, resources.max_time, self.queuing_sys_header,
                                      node_dict, no_default_header=self.no_default_header, verbose=verbose)
        except Exception:
            sys.stderr.write(f'System.submit call to Scheduler.submit failed for job id {id}, cleaning up remote dir {str(self.remote_rundir)}\n')
            self.run(['rm', '-r', str(job_remote_rundir)], verbose=verbose)
            raise

        if 'EXPYRE_TIMING_VERBOSE' in os.environ:
            sys.stderr.write(f'system {self.id} submit end {time.time()}\n')
        return r


    def get_remotes(self, local_dir, subdir_glob=None, verbose=False):
        """get data from directories of remotely running jobs

        Parameters
        ----------
        subdir_glob: str, list(str), default None
            only get subdirectories that much one or more globs
        """
        if subdir_glob is None:
            subdir_glob = '/*'
        elif isinstance(subdir_glob, str):
            subdir_glob = '/' + subdir_glob
        elif len(subdir_glob) == 1:
            subdir_glob = '/' + subdir_glob[0]
        else:
            subdir_glob = '/{' + ','.join(subdir_glob) + '}'

        subprocess_copy(self.remote_rundir + subdir_glob, local_dir, from_host=self.host,
                        remsh_cmd=self.remsh_cmd, verbose=verbose)



    def clean_rundir(self, stage_dir, filenames, dry_run=False, verbose=False):
        """clean a remote stage directory

        Parameters
        ----------
        stage_dir: str | Path
            local stage directory path
        files: list(str) or None
            list of files to replaced with 'CLEANED', or wipe entire directory if None
        verbose: bool, default False
            verbose output
        """
        job_remote_rundir = self._job_remote_rundir(Path(stage_dir))

        if filenames is not None:
            filenames = ['"' + filename + '"' for filename in filenames]
            self.run(['bash'],
                     script=(f'for f in {" ".join(filenames)}; do\n'
                             f'    ff={job_remote_rundir}/$f\n'
                             f'    if [ -f $ff ]; then\n'
                             f'        echo "CLEANED" > $ff\n'
                             f'    fi\n'
                             f'done\n'), dry_run=dry_run, verbose=verbose)
        else:
            self.run(['rm', '-rf', str(job_remote_rundir)], dry_run=dry_run, verbose=verbose)


    def __str__(self):
        s = f'System: host {self.host} rundir {self.remote_rundir} scheduler {type(self.scheduler).__name__}\n'
        if self.partitions is not None:
            s += ' ' + ' '.join(self.partitions.keys())
        return s
