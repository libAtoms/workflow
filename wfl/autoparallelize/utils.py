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

from ase.atoms import Atoms

from ..configset import ConfigSet
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
        environment var to get information from if not present in `remote_info` argument

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
                warnings.warn(f'remote_info "{remote_info}" from WFL_EXPYRE_INFO has whitespace, but not parseable as '
                              f'JSON/YAML with error {exc}')
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


def items_inputs_generator(iterable, num_inputs_per_group, rng):
    """Returns generator that returns tuples consisting of items, and associated data

    Parameters
    ----------
    iterable: iterable
        input quantities (often of type ase.atoms.Atoms)
    num_inputs_per_group: int
        number of inputs that will be included in each group
    rng: numpy.random.Generator or None
        rng to generate rngs for each item

    Returns
    -------
    generator that returns a sequence of items, each a tuple (item, item_i, item's _ConfigSet_loc, unique rng)
        (NOTE: _ConfigSet_loc is None unless item is ase.atoms.Atoms, rng is None unless rng is provided)
    """
    def _get_loc(item):
        loc = (item.info.get("_ConfigSet_loc") if isinstance(item, Atoms) else
              (item._enclosing_loc if isinstance(item, ConfigSet) else None))
        return loc

    return grouper(num_inputs_per_group,
                   ((item, item_i, _get_loc(item),
                     rng.spawn(1)[0] if rng is not None else None) for item_i, item in enumerate(iterable)))


def set_autopara_per_item_info(kwargs, op, inherited_per_item_info, rng_list, item_i_list):
    """Set some per-config information

    Parameters
    ----------
    kwargs: dict
        keyword args of op
    op: callable
        operation function
    inherited_per_item_info: list(dict)
        list of per-item info dicts that needs to be split up to these particular items
    rng_list: list(numpy.random.Generator) or list(None)
        rng (unique) for each item
    item_i_list: int
        list of sequence numbers for the items that these per-info items correspond to
    """
    if "_autopara_per_item_info" not in inspect.signature(op).parameters:
        return

    if inherited_per_item_info is not None:
        # divide up previous set
        kwargs["_autopara_per_item_info"] = [inherited_per_item_info[item_i] for item_i in item_i_list]
        return

    # create new autopara_per_item_info
    kwargs["_autopara_per_item_info"] = [{"item_i": item_i} for item_i in item_i_list]
    if rng_list[0] is not None:
        for item_info, item_rng in zip(kwargs["_autopara_per_item_info"], rng_list):
            item_info["rng"] = item_rng
