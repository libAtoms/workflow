import re

from ..subprocess import subprocess_run
from ..units import time_to_HMS
from .. import util

from .base import Scheduler


class Slurm(Scheduler):
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
        self.hold_command = ['scontrol', 'hold']
        self.release_command = ['scontrol', 'release']
        self.cancel_command = ['scancel']
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
            header.append('#SBATCH --job-name={id}')
            header.append('#SBATCH --partition={partition}')
            header.append('#SBATCH --time={max_time}')
            header.append('#SBATCH --output=job.{id}.stdout')
            header.append('#SBATCH --error=job.{id}.stderr')

        # set EXPYRE_NCORES_PER_NODE using scheduler-specific info, to support jobs
        # that do not know exact node type at submit time.  All related quantities
        # in node_dict are set based on this one by superclass Scheduler static method
        pre_commands = ['if [ ! -z $SLURM_TASKS_PER_NODE ]; then',
                        '    if echo "${{SLURM_TASKS_PER_NODE}}"| grep -q ","; then',
                        '        echo "Using only first part of hetereogeneous tasks per node spec ${{SLURM_TASKS_PER_NODE}}"',
                        '    fi',
                        '    export EXPYRE_NCORES_PER_NODE=$(echo $SLURM_TASKS_PER_NODE | sed "s/(.*//")',
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

        submit_args = ['cd', remote_dir, '&&', 'cat', '>', 'job.script.slurm',
                       '&&', 'sbatch', 'job.script.slurm']

        stdout, stderr = subprocess_run(self.host, args=submit_args, script=script, remsh_cmd=self.remsh_cmd, verbose=verbose)

        # parse stdout for remote job id
        remote_id = None
        for line in stdout.splitlines():
            m = re.match(r'^Submitted\s+batch\s+job\s+(\S+)', line.strip())
            if m is not None:
                if remote_id is not None:
                    raise RuntimeError('More than one line matches "Submitted batch job "')
                remote_id = m.group(1)
        if remote_id is None:
            raise RuntimeError('No line matches "Submitted batch job "')

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

        stdout, stderr = subprocess_run(self.host,
            ['squeue', '--user', '$LOGNAME', '--noheader', '-O', 'jobid:20,state:30,reason:200'],
            remsh_cmd=self.remsh_cmd, verbose=verbose)

        id_status_reasons = [line.strip().split(maxsplit=2) for line in stdout.splitlines()
                             if line.split()[0] in remote_ids]

        out = {}
        for id_status_reason in id_status_reasons:
            try:
                (id, status, reason) = id_status_reason
            except Exception:
                raise ValueError(f'failed to parse id_status_reason {id_status_reason}')

            if status in ['RUNNING', 'COMPLETING']:
                status = 'running'
            elif status == 'PENDING':
                if 'held' in reason.lower():
                    status = 'held'
                else:
                    status = 'queued'
            elif status == 'COMPLETED':
                status = 'done'
            elif 'fail' in status.lower():
                status = 'failed'
            elif status == 'TIMEOUT':
                status = 'timeout'
            else:
                status = 'other'
            out[id] = status

        for id in remote_ids:
            if id not in out:
                out[id] = 'done'
        return out
