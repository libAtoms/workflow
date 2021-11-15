"""functions dealing with running subprocesses, optionally on remote machines, handling
quoting/shell escping, encoding/decoding, environment, and making sure that optional
arguments needed for things like kerberized-ssh are set correctly.  """

import sys
import os
import subprocess
import re
import warnings
import time
import shlex

from pathlib import Path


class FailedSubprocessWarning(RuntimeWarning):
    pass


warnings.filterwarnings('always', category=FailedSubprocessWarning)


def _my_warn(msg):
    if 'pytest' in sys.modules:
        # also print to stderr since pytest shows all warnings
        # separately, and it's harder to use them to debug
        sys.stderr.write(msg + '\n')
        sys.stderr.flush()
    warnings.warn(msg, category=FailedSubprocessWarning)


def _optionally_remote_args(args, shell, host, remsh_cmd, in_dir='_HOME_'):
    """Convert args array into a shell call, optionally preceded by an ssh command

    Parameters
    ----------
    args: list(str)
        arguments to run
    shell: str
        shell to use
    host: str
        [username@]host to pass to ssh, None to run locally but act like ssh, i.e. start in $HOME,
    remsh_cmd: str, list(str)
        remote shell command (usually ssh)
    in_dir: str, default _HOME_
        directory to cd into before running args, _HOME_ for home dir, _PWD_ for python current working directory (only for host == None)

    Returns
    -------
    args list(str)
    """
    # make args appropriate for passing to subprocess.Popen, either directly
    # or preceded by an ssh command, with the shell as an explicit arg
    if isinstance(remsh_cmd, str):
        remsh_cmd = remsh_cmd.split()

    # NOTE: quoting below may be bash specific

    if in_dir == '_PWD_':
        assert host is None

    # get ready to pass args into bash as a single command:
    # join args into a single string, backslash escaping some chars in it, namely: ', ", space, (, )
    # start with "cd $HOME" so that whether or not there's an ssh, operations start relative to home dir
    cmd_str = ''
    if in_dir == '_HOME_':
        cmd_str = 'cd $HOME'
    elif in_dir != '_PWD_':
        cmd_str = 'cd ' + in_dir
    if len(args) > 0:
        if len(cmd_str) > 0:
            cmd_str += ' && '
        cmd_str += ' '.join([re.sub(r'([\'" \(\)])', r'\\\1', arg) for arg in args])
    args = shell.split() + [cmd_str]
    if host is not None:
        # pass remote command to an ssh command, in single quotes
        # args[0:-1] are shell itself and its flags, args[-1] is the command to run
        #     which needs to be single quoted
        args = remsh_cmd + [host] + args[0:-1] + ["'" + args[-1] + "'"]

    return args


def subprocess_run(host, args, script=None, shell='bash -lc', remsh_cmd=None, retry=None, in_dir='_HOME_', dry_run=False, verbose=False):
    """run a subprocess, optionally via ssh on a remote machine.  Raises RuntimeError for non-zero
    return status.

    Parameters
    ----------
    host: str
        [username]@machine.domain, or None for a locally run process
    args: list(str)
        arguments to run, starting with command and followed by its command line args
    script: str, default None
        text to write to process's standard input
    shell: str, default 'bash -lc'
        shell to use, including any flags necessary for it to interpret the next argument
        as the commands to run (-c for bash)
    remsh_command: str | list(str), default env var EXPYRE_RSH or 'ssh'
        command to start on remote host, usually ssh
    retry: (int, int), default env var EXPYRE_RETRY.split() or (2, 5)
        number of times to retry and number of seconds to wait between each trial
    in_dir: str, default _HOME_
        directory to cd into before running args, _HOME_ for home dir, _PWD_ for python current working directory (only for host == None)
    verbose: bool, default False
        verbose output

    Returns
    -------
    stdout, stderr: output and error of subprocess, as strings (bytes.decode())
    """
    if remsh_cmd is None:
        remsh_cmd = os.environ.get('EXPYRE_RSH', 'ssh')
    if retry is None:
        if 'EXPYRE_RETRY' in os.environ:
            retry = tuple([int(_ii) for _ii in os.environ['EXPYRE_RETRY'].strip().split()])
        else:
            retry = (3, 5)

    # always run at least once, and wait a valid (>= 0) amount of time
    retry = (max(retry[0], 1), max(retry[1], 0))

    args = _optionally_remote_args(args, shell, host, remsh_cmd, in_dir)

    if verbose:
        if dry_run:
            print('DRY-RUN COMMAND:')
        else:
            print('RUNNING COMMAND:')
        print(' '.join([shlex.quote(arg) for arg in args]))
        if script is not None:
            print('SCRIPT:')
            print(script.rstrip())

    if script is not None:
        script = script.encode()

    if dry_run:
        return args, script

    for i_try in range(retry[0]):
        try:
            p = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE, stdin=subprocess.PIPE, close_fds=False, env=os.environ)
            stdout, stderr = p.communicate(script)

            if p.returncode != 0:
                raise RuntimeError(f'Failed to run command "{" ".join(args)}" with err {stderr.decode()}')

            # success
            if i_try > 0:
                _my_warn(f'Succeeded to run "{" ".join(args)}" on attempt {i_try} after failure(s), trying again')
            break
        except Exception:
            if i_try == retry[0]-1:
                # last try
                _my_warn(f'Failed to run "{" ".join(args)}" on attempt {i_try} for the last time, giving up')

                # failed last chance
                raise
            _my_warn(f'Failed to run "{" ".join(args)}" on attempt {i_try}, trying again')

        time.sleep(retry[1])

    if verbose:
        print('GOT STDOUT:')
        print(stdout.decode())
        print('GOT STDERR:')
        print(stderr.decode())

    return stdout.decode(), stderr.decode()


def subprocess_copy(from_files, to_file, from_host='_LOCAL_', to_host='_LOCAL_',
                    rcp_args='-a', rcp_cmd='rsync', remsh_cmd=None, retry=None, remsh_flags='-e', verbose=False, dry_run=False):
    """Run a remote copy (e.g. rsync) in a subprocess, optionally to/from remote machine.  Exactly one
    machine has to be specified, and relative paths on that machine are relative to its home dir, like
    rsync.  If the specified machine is None the copy is local, but relative paths are still relative to
    home dir.

    Raises RuntimeError for non-zero return status.

    Parameters
    ----------
    from_files: str, Path, list(str), list(Path)
        one or more files/directories to copy from (not including 'user@host:' part)
    to_file: str, Path
        _one_ file/directory to copy to (not including 'user@host:' part)
    from_host: str, optional
        [username@]host.domain to copy from (no ":"), mutually exclusive with to_host, one is required.
        If None, use a local dir, but make relative paths relative to $HOME instead of to $PWD (like rsync with a host)
    to_host: str, optional
        [username@]host.domain to copy to (no ":"), mutually exclusive with from_host, one is required.
        If None, use a local dir, but make relative paths relative to $HOME instead of to $PWD (like rsync with a host)
    rcp_args: str, default '-a'
        non-filename arguments to remote copy command
    rcp_cmd: str, default 'rsync'
        command to do copy
    remsh_cmd: str, default EXPYRE_RSH env var or ssh
        arguments to set shell command for rcp_cmd to use
    retry: optional
        passed as retry argument to subprocess_run
    remsh_flags: str, default '-e'
        flag to prefix to remsh_cmd when calling rcp_cmd
    verbose: bool, default False
        verbose output
    """
    # exactly one of from_host, to_host must be provided
    if sum([from_host == '_LOCAL_', to_host == '_LOCAL_']) != 1:
        raise RuntimeError(f'Exactly one of source host "{from_host}" or target host "{to_host}" must passed in')

    if remsh_cmd is None:
        remsh_cmd = os.environ.get('EXPYRE_RSH', 'ssh')
    rcp_args = remsh_flags + ' ' + remsh_cmd + ' ' + rcp_args

    # make from_files plain str or Path into list
    if isinstance(from_files, str) or isinstance(from_files, Path):
        from_files = [from_files]

    if from_host is None:
        # local dir, but relative is relative to home dir rather than current dir, like rsync
        abs_from_files = []
        for f in from_files:
            if not Path(f).is_absolute():
                f = Path.home() / f
            abs_from_files.append(str(f))
    else:
        abs_from_files = from_files
    if from_host is None or from_host == '_LOCAL_':
        # make into string that can be prepended
        from_host = ''

    if to_host is None and not Path(to_file).is_absolute():
        # local dir, but relative is relative to home dir rather than current dir, like rsync
        abs_to_file = Path.home() / to_file
    else:
        abs_to_file = to_file
    if to_host is None or to_host == '_LOCAL_':
        # make into string that can be prepended
        to_host = ''

    # add ':' after non-blank host
    if len(from_host) > 0:
        from_host += ':'
    if len(to_host) > 0:
        to_host += ':'

    # prepend host parts to from_files and to_file
    abs_from_files = [from_host + str(f) for f in abs_from_files]
    abs_to_file = to_host + str(abs_to_file)

    # do copy (or dry run)
    retval = subprocess_run(None, [rcp_cmd] + rcp_args.split() + abs_from_files + [abs_to_file], retry=retry, in_dir='_PWD_', dry_run=dry_run, verbose=verbose)
    if dry_run:
        return retval
