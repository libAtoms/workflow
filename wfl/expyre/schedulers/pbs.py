import re

from ..subprocess import subprocess_run
from ..units import time_to_HMS
from .. import util

from .base import Scheduler


class PBS(Scheduler):
    def __init__(self, host, remsh_cmd=None):
        """Create Slurm object
        Parameters
        ----------
        host: str
            username and host for ssh/rsync username@machine.fqdn
        remsh_cmd: str, default EXPYRE_RSH env var or 'ssh'
            remote shell command to use
        """
        self.host = host
        self.hold_command = ['qhold']
        self.release_command = ['qrls']
        self.cancel_command = ['qdel']
        self.remsh_cmd = util.remsh_cmd(remsh_cmd)


    def submit(self, id, remote_dir, partition, commands, max_time, header, node_dict, no_default_header=False, verbose=False):
        """Submit a job on a remote machine
        Parameters
        ----------
        id: str
            unique job id (local)
        remote_dir: str
            remote directory where files have already been prepared and job will run
        partition: str
            partition (or queue or node type)
        commands: list(str)
            list of commands to run in script
        max_time: int
            time in seconds to run
        header: list(str)
            list of header directives, not including walltime specific directive
        node_dict: dict
            properties related to node selection.
            Fields: nnodes, tot_ntasks, tot_ncores, ncores_per_node, ppn, id, max_time, partition
        no_default_header: bool, default False
            do not add normal header fields, only use what's passed in in "header"

        Returns
        -------
        str remote job id
        """
        node_dict = node_dict.copy()

        # Make sure that there are no '=' in id
        # (e.g. produced by base64.urlsafe encoded argument hashes),
        # since those are rejected for "#PBS -N" despite the claim in the docs that
        # any printable non-whitespace character is OK:
        #     http://docs.adaptivecomputing.com/torque/4-0-2/Content/topics/commands/qsub.htm
        #
        # WARNING: since this is happening in scheduler-specific code, it will make stdout/stderr
        # filenames dependent on scheduler, so they cannot be relied on.  Would need to move
        # sanitization outside to prevent this, but it'd have to be superset of everything needed
        # for every scheduler.
        node_dict['id'] = id.replace('=', 'EQ')

        node_dict['max_time'] = time_to_HMS(max_time)
        node_dict['partition'] = partition

        header = header.copy()
        if not no_default_header:
            # Make sure that first characer is alphabetic
            # Let's hope there aren't length limitations anymore
            header.append('#PBS -N N_{id}')
            header.append('#PBS -q {partition}')
            header.append('#PBS -l walltime={max_time}')
            header.append('#PBS -o job.{id}.stdout')
            header.append('#PBS -e job.{id}.stderr')
            header.append('#PBS -S /bin/bash')
            header.append('#PBS -r n')

        # set EXPYRE_NCORES_PER_NODE using scheduler-specific info, to support jobs
        # that do not know exact node type at submit time.  All related quantities
        # in node_dict are set based on this one by superclass Scheduler static method
        pre_commands = ['if [ ! -z $PBS_NUM_PPN ]; then',
                        '    export EXPYRE_NCORES_PER_NODE=$PBS_NUM_PPN',
                        'elif [ ! -z $PBS_NODEFILE ]; then',
                        '    export EXPYRE_NCORES_PER_NODE=$(sort -k1 $PBS_NODEFILE | uniq -c | head -1 | awk \'{{print $1}}\')',
                        'else',
                       f'    export EXPYRE_NCORES_PER_NODE={node_dict["ntasks_per_node"]}',
                        'fi'
                       ] + Scheduler.node_dict_env_var_commands(node_dict)
        pre_commands = [l.format(**node_dict) for l in pre_commands]

        # add "cd remote_dir" before any other command
        if remote_dir.startswith('/'):
            pre_commands.append(f'cd {remote_dir}')
        else:
            pre_commands.append(f'cd ${{HOME}}/{remote_dir}')

        commands = pre_commands + commands

        script = '#!/bin/bash -l\n'
        script += '\n'.join([line.rstrip().format(**node_dict) for line in header]) + '\n'
        script += '\n' + '\n'.join([line.rstrip() for line in commands]) + '\n'

        submit_args = ['cd', remote_dir, '&&', 'cat', '>', 'job.script.pbs',
                       '&&', 'qsub', 'job.script.pbs']

        stdout, stderr = subprocess_run(self.host, args=submit_args, script=script, remsh_cmd=self.remsh_cmd, verbose=verbose)

        # parse stdout for remote job id
        if len(stdout.splitlines()) != 1:
            raise RuntimeError('More than one line in qsub output')
        remote_id = stdout.strip()
        if len(remote_id) == 0:
            raise RuntimeError('Empty output from qsub')

        return remote_id


    def status(self, remote_ids, verbose=False):
        """determine status of remote jobs
        Parameters
        ----------
        remote_ids: str, list(str)
            list of remote ids to check

        Returns
        -------
        dict { str remote_id: str status},  status is one of :
                "queued", "held", "running",   "done", "failed", "timeout", "other"
            all remote ids passed in are guaranteed to be keys in dict, specific jobs that are
            not listed by queueing system have status "done"
        """
        if isinstance(remote_ids, str):
            remote_ids = [remote_ids]

        # -w to make fields wide and less likely to truncate jobid
        # -x for historical data it should never say "Job has finished", but rather use same format for all jobs
        # -a for all jobs, most useful version that works with -w, hence need for grep USER
        stdout, stderr = subprocess_run(self.host,
            ['qstat', '-w', '-x', '-a', '|', 'grep', ' $USER '],
            remsh_cmd=self.remsh_cmd, verbose=verbose)

        lines = stdout.splitlines()
        # parse id and status from format
        # (id, _user, _queue, _jobname, _sessid, _nds, _tsk, _mem, _timereq, status, _timeelap) = l.strip().split()
        id_status = [(line.strip().split()[0], line.strip().split()[9]) for line in lines
                     if line.strip().split()[0] in remote_ids]
        out = {}
        for id, status in id_status:
            if status in ['R', 'E']:
                status = 'running'
            elif status == 'Q':
                status = 'queued'
            elif status == 'H':
                status = 'held'
            elif status == 'F':
                status = 'done'
            else:
                status = 'other'

            out[id] = status

        for id in remote_ids:
            if id not in out:
                out[id] = 'done'

        return out
