from ..subprocess import subprocess_run
from .. import util


class Scheduler:
    def __init__(self, host, remsh_cmd=None):
        """Create Scheduler object.  [NEED MORE INFO ABOUT HOW SCRIPTS WILL BE SET UP, THEIR ENVIRONMENT, ETC]
        Parameters
        ----------
        host: str
            username and host for ssh/rsync username@machine.fqdn
        """
        self.host = host
        self.hold_command = None
        self.release_command = None
        self.cancel_command = None
        self.remsh_cmd = util.remsh_cmd(remsh_cmd)


    def submit(self, id, remote_dir, partition, commands, max_time, header, node_dict, no_default_header=False, verbose=False):
        raise RuntimeError('Not implemented')


    def status(self, remote_ids, verbose=False):
        raise RuntimeError('Not implemented')


    def hold(self, remote_ids, verbose=False):
        """hold remote job
        Parameters:
        remote_ids: str, list(str)
            remote ids of jobs to hold
        """
        if isinstance(remote_ids, str):
            remote_ids = [remote_ids]

        subprocess_run(self.host, args=self.hold_command + remote_ids, remsh_cmd=self.remsh_cmd, verbose=verbose)


    def release(self, remote_ids, verbose=False):
        """release remote job
        Parameters:
        remote_ids: str, list(str)
            remote ids of jobs to hold
        """
        if isinstance(remote_ids, str):
            remote_ids = [remote_ids]

        subprocess_run(self.host, args=self.release_command + remote_ids, remsh_cmd=self.remsh_cmd, verbose=verbose)


    def cancel(self, remote_ids, verbose=False):
        """cancel remote job
        Parameters:
        remote_ids: str, list(str)
            remote ids of jobs to hold
        """
        if isinstance(remote_ids, str):
            remote_ids = [remote_ids]

        subprocess_run(self.host, args=self.cancel_command + remote_ids, remsh_cmd=self.remsh_cmd, verbose=verbose)


    @staticmethod
    def node_dict_env_var_commands(node_dict):
        # set env vars for node_dict, with max flexibility in case only some are known at submit time

        # EXPYRE_NCORES_PER_NODE is defined by scheduler before these commands are run

        # ncores per task is alawys set by resources
        pre_commands = ['export EXPYRE_NCORES_PER_TASK={ncores_per_task}']

        # either nnodes or tot_ncores must be known at submit time, so compute each in terms of the other
        if node_dict['nnodes'] is None:
            pre_commands.append('export EXPYRE_NNODES=$(( {tot_ncores} / $EXPYRE_NCORES_PER_NODE ))')
        else:
            pre_commands.append('export EXPYRE_NNODES={nnodes}')
        if node_dict['tot_ncores'] is None:
            pre_commands.append('export EXPYRE_TOT_NCORES=$(( {nnodes} * $EXPYRE_NCORES_PER_NODE ))')
        else:
            pre_commands.append('export EXPYRE_TOT_NCORES={tot_ncores}')

        # compute from ncores_per_node and ncores_per_task
        if node_dict['ntasks_per_node'] is None:
            pre_commands.append('export EXPYRE_NTASKS_PER_NODE=$(( $EXPYRE_NCORES_PER_NODE / $EXPYRE_NCORES_PER_TASK ))')
        else:
            pre_commands.append('export EXPYRE_NTASKS_PER_NODE={ntasks_per_node}')

        # compute from nnodes and ntasks_per_node
        if node_dict['tot_ntasks'] is None:
            pre_commands.append('export EXPYRE_TOT_NTASKS=$(( $EXPYRE_NNODES * $EXPYRE_NTASKS_PER_NODE ))')
        else:
            pre_commands.append('export EXPYRE_TOT_NTASKS={tot_ntasks}')

        return pre_commands
