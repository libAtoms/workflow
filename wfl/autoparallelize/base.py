import sys
import os
import json
import warnings
import traceback as tb
import inspect
import functools
import re

from wfl.configset import OutputSpec
from .pool import do_in_pool
from .remote import do_remotely

iloop_docstring_params_pre = (
"""
inputs: iterable({iterable_type})
    input quantities of type {iterable_type}
outputs: OutputSpec or None
    where to write output atomic configs, or None for no output (i.e. only side-effects)
""")

iloop_docstring_params_post = (
"""
ITERABLE_LOOP_RELATED:

    - num_python_subprocesses: int, default os.environ['WFL_NUM_PYTHON_SUBPROCESSES']
      number of processes to parallelize over, 0 for running in serial
    - num_inputs_per_python_subprocess: int, default 1 (kwargs only)
      number of items from iterable to pass to each invocation of operation (pass to iterable_loop())
    - skip_failed: bool, default True
      skip function calls that return None
    - remote_info: RemoteInfo, default content of env var WFL_EXPYRE_INFO
      information for running on remote machine.  If None, use WFL_EXPYRE_INFO env var, as
      json file if string, as RemoteInfo kwargs dict if keys include sys_name, or as dict of
      RemoteInfo kwrgs with keys that match end of stack trace with function names separated by '.'.
    - remote_label: str, default None
      remote_label to use for operation, to match to remote_info dict keys.  If none, use calling routine
      filename '::' calling function (pass to iterable_loop())
""")

iloop_docstring_returns = (
"""
Returns
-------
    co: ConfigSet
        output configs
""")

def iloop_docstring(orig_docstring, input_iterable_type):
    output_docstring = ""
    got_pre = False
    lines = orig_docstring.splitlines(True)
    on_parameter_line = False
    for li in range(len(lines)):
        prev_line_match = re.match(r'^(\s*)Parameters\s*[:]?\s*$', lines[li-1])
        if (li >= 1 and re.match(r'^\s*[-]*\s*$', lines[li]) and prev_line_match):
            on_parameter_line = True
            output_docstring += lines[li]
            got_pre = True
            section_init_space = prev_line_match.group(1)
        elif on_parameter_line:
            m = re.match(r'(\s*)', lines[li])
            param_init_space = m.group(1)
            output_docstring += ''.join([param_init_space + l for l in iloop_docstring_params_pre.format(iterable_type=input_iterable_type).splitlines(True)])
            on_parameter_line = False
            output_docstring += lines[li]
        else:
            output_docstring += lines[li]

    output_docstring += ''.join([param_init_space + l for l in iloop_docstring_params_post.splitlines(True)])

    lines = orig_docstring.splitlines(True)
    if not re.match(r'^\s*$', lines[-1]):
        output_docstring += "\n"
    if output_docstring[-1] != "\n":
        output_docstring += "\n"

    output_docstring += ''.join([section_init_space + l for l in iloop_docstring_returns.splitlines(True)])

    return output_docstring


def iloop(func, *args,
          def_num_python_subprocesses=None, def_num_inputs_per_python_subprocess=1, iterable_arg=0, def_skip_failed=True,
          initializer=None, initargs=None, def_remote_info=None, def_remote_label=None, hash_ignore=[], **kwargs):
    """functools.partial-based decorator (using ideas in topic 4 of
    https://pythonicthoughtssnippets.github.io/2020/08/09/PTS13-rethinking-python-decorators.html).
    Works OK and can be pickled, but ugly handling of docstring, and no handling of function signature.

    Use by defining operation `op` which takes an input iterable and returns list of configs, and _after_ do

    .. code-block:: python

        def autopara_op(*args, **kwargs):
            f = functools.partial(iloop, op, [ iloop_keyword_param_1=val, iloop_keyword_param_2=val, ... ] )
            return f(*args, **kwargs)
        autopara_op.doc = iloop_docstring(op.__doc__, "iterable_contents")


    The autoparallelized function can then be called with `autopara_op(inputs, outputs, args of op)`


    Parameters
    ----------
    func: function
        function to wrap in iterable_loop()
    input_type: str
        type of input configs
    def_num_python_subprocesses: int, default os.environ['WFL_NUM_PYTHON_SUBPROCESSES']
        number of processes to parallelize over, 0 for running in serial
    def_num_inputs_per_python_subprocess: int, default 1
        default number of items from iterable to pass to each invocation of operation (pass to iterable_loop())
    iterable_arg: int or str, default 0
        positional argument or keyword argument to place iterable items in when calling op (pass to iterable_loop())
    def_skip_failed: bool, default True
        skip function calls that return None
    initializer: callable, default None
        function to call at beginning of each thread (pass to iterable_loop())
    initargs: list, default None
        positional arguments for initializer (pass to iterable_loop())
    def_remote_info: RemoteInfo, default content of env var WFL_EXPYRE_INFO
        information for running on remote machine.  If None, use WFL_EXPYRE_INFO env var, as
        json file if string, as RemoteInfo kwargs dict if keys include sys_name, or as dict of
        RemoteInfo kwrgs with keys that match end of stack trace with function names separated by '.'.
    def_remote_label: str, default None
        remote_label to use for operation, to match to remote_info dict keys.  If none, use calling routine filename '::' calling function (pass to iterable_loop())
    hash_ignore: list(str), default []
        arguments to ignore when doing remot executing and computing hash of function to determine
        if it's already done (pass to iterable_loop())

    Returns
    -------
    wrapped_func: function wrapped in autoparallelize via iloop
    """

    # copy kwargs and args so they can be modified for call to autoparallelize
    kwargs = kwargs.copy()
    args = list(args)
    if 'inputs' in kwargs:
        # inputs is keyword, outputs must be too, any positional args to func are unchanged
        inputs = kwargs.pop('inputs')
        outputs = kwargs.pop('outputs')
    else:
        # inputs is positional, remove it from args to func
        inputs = args.pop(0)
        if 'outputs' in kwargs:
            outputs = kwargs.pop('outputs')
        else:
            # outputs is also positional, remote it from func args as well
            outputs = args.pop(0)

    num_python_subprocesses = kwargs.pop('num_python_subprocesses', def_num_python_subprocesses)
    num_inputs_per_python_subprocess = kwargs.pop('num_inputs_per_python_subprocess', def_num_inputs_per_python_subprocess)
    skip_failed = kwargs.pop('skip_failed', def_skip_failed)
    remote_info = kwargs.pop('remote_info', def_remote_info)
    remote_label = kwargs.pop('remote_label', def_remote_label)

    return autoparallelize(num_python_subprocesses, num_inputs_per_python_subprocess, inputs, outputs, func, iterable_arg, skip_failed,
                         initializer, initargs, remote_info, remote_label, hash_ignore, *args, **kwargs)

# do we want to allow for ops that only take singletons, not iterables, as input, maybe with num_inputs_per_python_subprocess=0?
# that info would have to be passed down to _wrapped_autopara_wrappable so it passes a singleton rather than a list into op
#
# some ifs (int positional vs. str keyword) could be removed if we required that the iterable be passed into a kwarg.
def autoparallelize(num_python_subprocesses=None, num_inputs_per_python_subprocess=1, iterable=None, outputspec=None, op=None, iterable_arg=0, skip_failed=True,
                  initializer=None, initargs=None, remote_info=None, remote_label=None, hash_ignore=[], *args, **kwargs):
    """parallelize some operation over an iterable

    Parameters
    ----------
    num_python_subprocesses: int, default os.environ['WFL_NUM_PYTHON_SUBPROCESSES']
        number of processes to parallelize over, 0 for running in serial
    num_inputs_per_python_subprocess: int, default 1
        number of items from iterable to pass to kach invocation of operation
    iterable: iterable, default None
        iterable to loop over, often ConfigSet but could also be other things like range()
    outputspec: OutputSpec, default None
        object containing returned Atoms objects
    op: callable
        function to call with each chunk
    iterable_arg: int or str, default 0
        positional argument or keyword argument to place iterable items in when calling op
    skip_failed: bool, default True
        skip function calls that return None
    initializer: callable, default None
        function to call at beginning of each thread
    initargs: list, default None
        positional arguments for initializer
    remote_info: RemoteInfo, default content of env var WFL_EXPYRE_INFO
        information for running on remote machine.  If None, use WFL_EXPYRE_INFO env var, as
        json file if string, as RemoteInfo kwargs dict if keys include sys_name, or as dict of
        RemoteInfo kwrgs with keys that match end of stack trace with function names separated by '.'.
    remote_label: str, default None
        remote_label to use for operation, to match to remote_info dict keys.  If none, use calling routine filename '::' calling function
    hash_ignore: list(str), default []
        arguments to ignore when doing remot executing and computing hash of function to determine
        if it's already done
    args: list
        positional arguments to op
    kwargs: dict
        keyword arguments to op

    Returns
    -------
    ConfigSet containing returned configs if outputspec is not None, otherwise None
    """
    if initargs is None:
        initargs = []

    if remote_info is None and 'WFL_EXPYRE_INFO' in os.environ:
        try:
            remote_info = json.loads(os.environ['WFL_EXPYRE_INFO'])
        except Exception as exc:
            remote_info = os.environ['WFL_EXPYRE_INFO']
            if ' ' in remote_info:
                # if it's not JSON, it must be a filename, so presence of space is suspicious
                warnings.warn(f'remote_info from WFL_EXPYRE_INFO has whitespace, but not parseable as JSON with error {exc}')
        if isinstance(remote_info, str):
            # filename
            with open(remote_info) as fin:
                remote_info = json.load(fin)
        if 'sys_name' in remote_info:
            # remote_info directly in top level dict
            warnings.warn('WFL_EXPYRE_INFO appears to be a RemoteInfo kwargs, using directly')
        else:
            if remote_label is None:
                # no explicit remote_label for the remote run was passed into function, so
                # need to match end of stack trace to remote_info dict keys, here we
                # construct object to compare to
                # last stack item is always autoparallelize, so ignore it
                stack_remote_label = [fs[0] + '::' + fs[2] for fs in tb.extract_stack()[:-1]]
            else:
                stack_remote_label = []
            if len(stack_remote_label) > 0 and stack_remote_label[-1].endswith('base.py::iloop'):
                # replace iloop stack entry with one for desired function name
                stack_remote_label.pop()
            #DEBUG print("DEBUG stack_remote_label", stack_remote_label)
            match = False
            for ri_k in remote_info:
                ksplit = [sl.strip() for sl in ri_k.split(',')]
                # match dict key to remote_label if present, otherwise end of stack
                if ((remote_label is None and all([re.search(kk + '$', sl) for sl, kk in zip(stack_remote_label[-len(ksplit):], ksplit)])) or
                    (remote_label == ri_k)):
                    sys.stderr.write(f'WFL_EXPYRE_INFO matched key {ri_k} for remote_label {remote_label}\n')
                    remote_info = remote_info[ri_k]
                    match = True
                    break
            if not match:
                remote_info = None

    if isinstance(iterable_arg, int):
        assert len(args) >= iterable_arg
        # otherwise not enough args were provided

    if outputspec is not None:
        if not isinstance(outputspec, OutputSpec):
            raise RuntimeError('iterable_loop requires outputspec be None or OutputSpec')
        if outputspec.is_done():
            sys.stderr.write(f'Returning before {op} since output is done\n')
            return outputspec.to_ConfigSet()

    if remote_info is not None:
        out = do_remotely(remote_info, hash_ignore, num_inputs_per_python_subprocess, iterable, outputspec,
                          op, iterable_arg, skip_failed, initializer, initargs, args, kwargs)
    else:
        out = do_in_pool(num_python_subprocesses, num_inputs_per_python_subprocess, iterable, outputspec, op, iterable_arg,
                         skip_failed, initializer, initargs, args, kwargs)

    return out
