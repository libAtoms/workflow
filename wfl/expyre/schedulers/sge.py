import re

from ..subprocess import subprocess_run
from ..units import time_to_HMS
from .. import util

from .base import Scheduler


class SGE(Scheduler):
    def __init__(self, host, remsh_cmd=None):
        """Create SGE object
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

        node_dict['id'] = id
        node_dict['max_time'] = time_to_HMS(max_time)
        node_dict['partition'] = partition

        header = header.copy()
        if not no_default_header:
            # Make sure that first characer is alphabetic
            header.append('#$ -N N_{id}')
            header.append('#$ -q {partition}')
            header.append('#$ -l h_rt={max_time}')
            header.append('#$ -o job.{id}.stdout')
            header.append('#$ -e job.{id}.stderr')
            header.append('#$ -S /bin/bash')
            header.append('#$ -r n')
            header.append('#$ -cwd')

        # set EXPYRE_NCORES_PER_NODE using scheduler-specific info, to support jobs
        # that do not know exact node type at submit time.  All related quantities
        # in node_dict are set based on this one by superclass Scheduler static method
        # for now assuming jobs can only run on single node (e.g. on Womble)
        pre_commands = ['if [ ! -z $NSLOTS ] && [ ! -z $NHOSTS ]; then',
                        '    export EXPYRE_NCORES_PER_NODE=$(( ${{NSLOTS}} / ${{NHOSTS}} ))',
                        'else',
                        '    export EXPYRE_NCORES_PER_NODE={ntasks_per_node}',
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

        submit_args = ['cd', remote_dir, '&&', 'cat', '>', 'job.script.sge',
                       '&&', 'qsub', 'job.script.sge']

        stdout, stderr = subprocess_run(self.host, args=submit_args, script=script, remsh_cmd=self.remsh_cmd, verbose=verbose)

        # parse stdout for remote job id
        if len(stdout.splitlines()) != 1:
            raise RuntimeError('More than one line in qsub output')
        m = re.match(r'Your\s+job\s+(\d+)\s+\("\S+"\)\s+has\s+been\s+submitted', stdout.strip())
        if m is None:
            raise RuntimeError('Empty output from qsub')

        return m.group(1)


    def status(self, remote_ids, verbose=False):
        """determine status of remote jobs

        Parameters
        ----------
        remote_ids: str, list(str)
            list of remote ids to check

        Returns
        -------
        dict { str remote_id: str status},  status is one of :
                "queued", "held", "running",  "done", "failed", "timeout", "other"
            all remote ids passed in are guaranteed to be keys in dict, specific jobs that are
            not listed by queueing system have status "done"
        """
        if isinstance(remote_ids, str):
            remote_ids = [remote_ids]

        stdout, stderr = subprocess_run(self.host,['qstat'],
            remsh_cmd=self.remsh_cmd, verbose=verbose)

        # first two lines are header
        lines = stdout.splitlines()[2:]
        # parse id and status from format
        # (id, _priority, _jobname, _user, status, _sub_or_start_date,
        # _sub_or_start_time, [_queue], _nprocs) = l.strip().split()
        id_status = [(line.strip().split()[0], line.strip().split()[4]) for line in lines
                     if line.strip().split()[0] in remote_ids]
        out = {}
        for id, status in id_status:
            if status in ['t', 'r']:
                status = 'running'
            elif status == 'qw':
                status = 'queued'
            elif status == 'hqw':
                status = 'held'
            else:
                status = 'other'

            out[id] = status

        for id in remote_ids:
            if id not in out:
                out[id] = 'done'

        return out
