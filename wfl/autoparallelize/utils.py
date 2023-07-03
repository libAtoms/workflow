import sys
import os
import io
import yaml
import traceback as tb
import re
import warnings
import itertools
import inspect

import numpy as np

from .remoteinfo import RemoteInfo

def grouper(n, iterable):
    """iterator that goes over iterable in specified size groups

    Parameters
    ----------
    iterable: any iterable
        iterable to loop over
    n: int
        size of group in each returned tuple

    Returns
    -------
    sequence of tuples, with items from iterable, each of size n (or smaller if n items are not available)
    """
    it = iter(iterable)
    while True:
        chunk = tuple(itertools.islice(it, n))
        if not chunk:
            return
        yield chunk


def get_remote_info(remote_info, remote_label, env_var="WFL_EXPYRE_INFO"):
    """get remote_info dict from passed in dict, label, and/or env. var

    Parameters
    ----------

    remote_info: RemoteInfo, default content of env var WFL_EXPYRE_INFO
        information for running on remote machine.  If None, use WFL_EXPYRE_INFO env var, as
        json/yaml file if string, as RemoteInfo kwargs dict if keys include sys_name, or as dict of
        RemoteInfo kwrgs with keys that match end of stack trace with function names separated by '.'.
    remote_label: str, default None
        remote_label to use for operation, to match to remote_info dict keys.  If none, use calling routine filename '::' calling function
    env_var: str, default "WFL_EXPYRE_INFO"
        environment var to get information from if not present in `remote\_info` argument

    Returns
    -------
    remote_info: RemoteInfo or None
    """
    if remote_info is None and env_var in os.environ:
        try:
            env_var_stream = io.StringIO(os.environ[env_var])
            remote_info = yaml.safe_load(env_var_stream)
        except Exception as exc:
            remote_info = os.environ[env_var]
            if ' ' in remote_info:
                # if it's not JSON, it must be a filename, so presence of space is suspicious
                warnings.warn(f'remote_info "{remote_info}" from WFL_EXPYRE_INFO has whitespace, but not parseable as JSON/YAML with error {exc}')
        if isinstance(remote_info, str):
            # filename
            with open(remote_info) as fin:
                remote_info = yaml.safe_load(fin)
        if 'sys_name' in remote_info:
            # remote_info directly in top level dict
            warnings.warn(f'env var {env_var} appears to be a RemoteInfo kwargs, using directly')
        else:
            if remote_label is None:
                # no explicit remote_label for the remote run was passed into function, so
                # need to match end of stack trace to remote_info dict keys, here we
                # construct object to compare to
                # last stack item is always autoparallelize, so ignore it
                stack_remote_label = [fs[0] + '::' + fs[2] for fs in tb.extract_stack()[:-1]]
            else:
                stack_remote_label = []
            while len(stack_remote_label) > 0 and (stack_remote_label[-1].endswith('autoparallelize/base.py::autoparallelize') or
                                                   stack_remote_label[-1].endswith('autoparallelize/base.py::_autoparallelize_ll') or
                                                   stack_remote_label[-1].endswith('autoparallelize/utils.py::get_remote_info')):
                # replace autoparallelize stack entry with one for desired function name
                stack_remote_label.pop()
            #DEBUG print("DEBUG stack_remote_label", stack_remote_label)
            match = False
            for ri_k in remote_info:
                ksplit = [sl.strip() for sl in ri_k.split(',')]
                # match dict key to remote_label if present, otherwise end of stack
                if ((remote_label is None and all([re.search(kk + '$', sl) for sl, kk in zip(stack_remote_label[-len(ksplit):], ksplit)])) or
                    (remote_label == ri_k)):
                    sys.stderr.write(f'{env_var} matched key {ri_k} for remote_label {remote_label}\n')
                    remote_info = remote_info[ri_k]
                    match = True
                    break
            if not match:
                remote_info = None

    if isinstance(remote_info, dict):
        remote_info = RemoteInfo(**remote_info)

    return remote_info

def get_root_global_seed(kwargs, op, label):
    """get root global seed from kwargs

    See https://numpy.org/doc/stable/reference/random/parallel.html#sequence-of-integer-seeds

    Parameters
    ----------
    kwargs: dict
        operation keyword arguments
    op: callable
        operation function
    label: any
        label for warning if using random seed

    Returns
    -------
    root_global_seed: int seed with value kwargs["autopara_rng_seed"] or random int if value is None or not in kwargs, or None if "autopara_rng_seed" is not in op's signature
    """
    if "autopara_rng_seed" in inspect.signature(op).parameters:
        if kwargs.get("autopara_rng_seed") is None:
            root_global_seed = np.random.randint(2 ** 32 - 1)
            warnings.warn(f"Using random root seed {root_global_seed} for {label}")
        else:
            root_global_seed = kwargs["autopara_rng_seed"]
    else:
        root_global_seed = None

    return root_global_seed

def set_autopara_per_item_info(kwargs, op, root_global_seed, prev_per_item_info, item_i_list):
    """Set some per-config information based on a root global seed and list of sequence ids

    See https://numpy.org/doc/stable/reference/random/parallel.html#sequence-of-integer-seeds

    Parameters
    ----------
    kwargs: dict
        keyword args of op
    op: callable
        operation function
    root_global_seed: int or None
        root global seed, or None if no per-item rng seed is needed
    prev_per_item_info: list(dict)
        list of per-item info dicts that needs to be split up to these particular items
    item_i_list: int
        list of sequence numbers for the items that these per-info items correspond to
    """
    if "autopara_per_item_info" not in inspect.signature(op).parameters:
        return

    if prev_per_item_info is not None:
        # divide up previous set
        kwargs["autopara_per_item_info"] = [prev_per_item_info[item_i] for item_i in item_i_list]
    else:
        # create new autopara_per_item_info
        kwargs["autopara_per_item_info"] = [{"item_i": item_i} for item_i in item_i_list]
        if root_global_seed is not None:
            # add seeds if root seed is available
            for per_item_info in kwargs["autopara_per_item_info"]:
                per_item_info["rng_seed"] = [per_item_info["item_i"], root_global_seed]
