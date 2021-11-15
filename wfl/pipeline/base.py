import sys
import os
import json
import warnings
import traceback as tb
import re

from wfl.configset import ConfigSet_out
from .pool import do_in_pool
from .remote import do_remotely


# do we want to allow for ops that only take singletons, not iterables, as input, maybe with chunksize=0?
# that info would have to be passed down to _wrapped_op so it passes a singleton rather than a list into op
#
# some ifs (int positional vs. str keyword) could be removed if we required that the iterable be passed into a kwarg.
def iterable_loop(npool=None, chunksize=1, iterable=None, configset_out=None, op=None, iterable_arg=0, skip_failed=True,
                  initializer=None, initargs=None, remote_info=None, label=None, hash_ignore=[], *args, **kwargs):
    """parallelize some operation over an iterable
    Parameters
    ----------
    npool: int, default os.environ['WFL_AUTOPARA_NPOOL']
        number of processes to parallelize over, 0 for running in serial
    chunksize: int, default 1
        number of items from iterable to pass to kach invocation of operation
    iterable: iterable, default None
        iterable to loop over, often ConfigSet_in but could also be other things like range()
    configset_out: ConfigSet_out, defaulat None
        object containing returned Atoms objects
    op: callable
        function to call with each chunk
    iterable_arg: itr or str, default 0
        positional argument or keyword argument to place iterable items in when calling op
    skip_failed: bool, default True
        skip function calls that return None
    initializer: callable, default None
        function to call at beginning of each thread
    initargs: list, default None
        positional arguments for initializer
    remote_info: RemoteInfo, default content of env var WFL_AUTOPARA_REMOTEINFO
        information for running on remote machine.  If None, use WFL_AUTOPARA_REMOTEINFO env var, as
        json file if string, as RemoteInfo kwargs dict if keys include sys_name, or as dict of
        RemoteInfo kwrgs with keys that match end of stack trace with function names separated by '.'.
    label: str, default None
        label to use for operation, to match to remote_info dict keys.  If none, use calling routine filename '::' calling function
    args: list
        positional arguments to op
    kwargs: dict
        keyword arguments to op

    Returns
    -------
    ConfigSet_in containing returned configs if configset_out is not None, otherwise None
    """
    if initargs is None:
        initargs = []

    if remote_info is None and 'WFL_AUTOPARA_REMOTEINFO' in os.environ:
        try:
            remote_info = json.loads(os.environ['WFL_AUTOPARA_REMOTEINFO'])
        except Exception as exc:
            remote_info = os.environ['WFL_AUTOPARA_REMOTEINFO']
            if ' ' in remote_info:
                # if it's not JSON, it must be a filename, so presence of space is suspicious
                warnings.warn(f'remote_info from WFL_AUTOPARA_REMOTEINFO has whitespace, but not parseable as JSON with error {exc}')
        if isinstance(remote_info, str):
            # filename
            with open(remote_info) as fin:
                remote_info = json.load(fin)
        if 'sys_name' in remote_info:
            # remote_info directly in top level dict
            warnings.warn('WFL_AUTOPARA_REMOTEINFO appears to be a RemoteInfo kwargs, using directly')
        else:
            if label is None:
                # no explicit label for the remote run was passed into function, so
                # need to match end of stack trace to remote_info dict keys, here we
                # construct object to compare to
                stack_label = [fs[0] + '::' + fs[2] for fs in tb.extract_stack()[:-1]]
            match = False
            for ri_k in remote_info:
                ksplit = [sl.strip() for sl in ri_k.split(',')]
                if (label is not None and label == ri_k) or all([re.search(kk + '$', sl) for sl, kk in zip(stack_label[-len(ksplit):], ksplit)]):
                    sys.stderr.write(f'WFL_AUTOPARA_REMOTEINFO matched key {ri_k} for label {label}\n')
                    remote_info = remote_info[ri_k]
                    match = True
                    break
            if not match:
                remote_info = None

    if isinstance(iterable_arg, int):
        assert len(args) >= iterable_arg
        # otherwise not enough args were provided

    if configset_out is not None:
        if not isinstance(configset_out, ConfigSet_out):
            raise RuntimeError('iterable_loop requires configset_out be None or ConfigSet_out')
        if configset_out.is_done():
            sys.stderr.write(f'Returning before {op} since output is done\n')
            return configset_out.to_ConfigSet_in()

    if remote_info is not None:
        out = do_remotely(remote_info, hash_ignore, chunksize, iterable, configset_out,
                          op, iterable_arg, skip_failed, initializer, initargs, args, kwargs)
    else:
        out = do_in_pool(npool, chunksize, iterable, configset_out, op, iterable_arg,
                         skip_failed, initializer, initargs, args, kwargs)

    return out
