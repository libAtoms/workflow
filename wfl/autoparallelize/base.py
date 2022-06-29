import sys
import os
import json
import warnings
import traceback as tb
import re

from wfl.configset import OutputSpec
from .pool import do_in_pool
from .remote import do_remotely

# NOTES from a previous implementation that does not work with sphinx docstring parsing
# may not be exactly correct, but can definitely be made to work (except sphinx)
#
# note that it is possible to define iloop so that the decorator functionality can be
# done with
#
#     parallelized_op = iloop(op, "parallelized_op", "Atoms",. [iloop_arg_1 = ... ])
#
# this is done by renaming the current "iloop" to "_iloop_wrapper", and defining a new "iloop"
# as something like
#
#     def iloop(op, op_name, input_iterator_contents, **iloop_kwargs):
#         f = functools.partial(_iloop_wrapper, op, **iloop_kwargs)
#         f.__name__ = op_name
#         f.__doc__ = iloop_docstring(op.__doc__, input_iterator_contents)
#         return f
#
# it also requires that the code parsing the stack trace to match up the RemoteInfo dict
# detects the presence of "iloop" in the stack trace, and instead uses
#
#     inspect.getfile(op) + "::" + op.__name__
#
# instead of the stack trace file and function
#
# sphinx docstring parsing cannot handle this, though, because the module associated with
# the "parallelized_op" symbol is that of "iloop", rather than the python file in which
# it is actually defined.  As a result, sphinx assumes it's just some imported symbol and does
# not include its docstring.  There does not appear to be any way to override that on a
# per-symbol basis in current sphinx versions.

iloop_docstring_params_pre = (
"""inputs: iterable({iterable_type})
    input quantities of type {iterable_type}
outputs: OutputSpec or None
    where to write output atomic configs, or None for no output (i.e. only side-effects)
""")

iloop_docstring_params_post = (
"""ITERABLE_LOOP_RELATED:

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
"""Returns
-------
    co: ConfigSet
        output configs
""")

def iloop_docstring(orig_docstring, input_iterable_type):
    output_docstring = ""
    lines = orig_docstring.splitlines(True)
    prev_line_was_parameters_section = False
    for li in range(len(lines)):
        prev_line_match = re.match(r'^(\s*)Parameters\s*[:]?\s*$', lines[li-1])
        if (li >= 1 and re.match(r'^\s*[-]*\s*$', lines[li]) and prev_line_match):
            # set flag so that iloop_docstring_params_pre can be inserted starting on the next line
            prev_line_was_parameters_section = True
            # save line-initial space on parameter line to keep indentation consistent for Returns
            # section that will be inserted into docstring later
            section_init_space = prev_line_match.group(1)
        elif prev_line_was_parameters_section and not re.match(r'^\s*$', lines[li]):
            # save line-initial space from first post-Parameters line so that other parameters that will be 
            # inserted into docstring will have consistent indentation
            m = re.match(r'(\s*)', lines[li])
            param_init_space = m.group(1)
            # insert extra lines between Parameters section header and initial function parameters,
            # with consistent indentation
            output_docstring += ''.join([param_init_space + l for l in iloop_docstring_params_pre.format(iterable_type=input_iterable_type).splitlines(True)])
            # no longer need to insert iloop-related lines
            prev_line_was_parameters_section = False
        # save orig line
        output_docstring += lines[li]

    # make sure output_docstring has not extra blank lines at the end
    while re.match(r'^\s*$', output_docstring.splitlines()[-1]):
        output_docstring = "\n".join(output_docstring.splitlines()[:-1])
    # make sure docstring ends with carriage return
    if not output_docstring.endswith("\n"):
        output_docstring += "\n"

    # insert docstring lines for parameters that come _after_ function's real parameters
    output_docstring += '\n' + ''.join([param_init_space + l for l in iloop_docstring_params_post.splitlines(True)])

    # insert Returns section
    output_docstring += '\n' + ''.join([section_init_space + l for l in iloop_docstring_returns.splitlines(True)])

    return output_docstring


def iloop(func, *args,
          def_num_python_subprocesses=None, def_num_inputs_per_python_subprocess=1, iterable_arg=0, def_skip_failed=True,
          initializer=None, initargs=None, def_remote_info=None, def_remote_label=None, hash_ignore=[], **kwargs):
    """functools.partial-based decorator (using ideas in topic 4 of
    https://pythonicthoughtssnippets.github.io/2020/08/09/PTS13-rethinking-python-decorators.html).
    Works OK and can be pickled, but ugly handling of docstring, and no handling of function signature.

    Use by defining operation `op` which takes an input iterable and returns list of configs, and _after_ do

    .. code-block:: python

        def parallelized_op(*args, **kwargs):
            return iloop(op, *args, [ iloop_keyword_param_1=val, iloop_keyword_param_2=val, ... ], **kwargs )
        parallelized_op.doc = iloop_docstring(op.__doc__, "iterable_contents")

    The autoparallelized function can then be called with `parallelized_op(inputs, outputs, [args of op], [args of iloop])`


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
