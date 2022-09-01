import sys
import os
import json
import warnings
import re

from wfl.configset import ConfigSet, OutputSpec
from .pool import do_in_pool
from .remote import do_remotely
from .autoparainfo import AutoparaInfo
from .utils import get_remote_info

# NOTES from a previous implementation that does not work with sphinx docstring parsing
# may not be exactly correct, but can definitely be made to work (except sphinx)
#
# It is possible to define autoparallelize so that the decorator functionality can be done with
#
#     parallelized_op = autoparallelize(op, "parallelized_op", "Atoms",. [autoparallelize_arg_1 = ... ])
#
# using functools.partial-based decorator (with ideas from topic 4 of
# https://pythonicthoughtssnippets.github.io/2020/08/09/PTS13-rethinking-python-decorators.html).
# Works OK and can be pickled, but ugly handling of docstring, and no handling of function signature.
#
# this is done by renaming the current "autoparallelize" to "_autoparallelize_wrapper", and defining a new "autoparallelize"
# as something like
#
#     def autoparallelize(op, op_name, input_iterator_contents, **autoparallelize_kwargs):
#         f = functools.partial(_autoparallelize_wrapper, op, **autoparallelize_kwargs)
#         f.__name__ = op_name
#         f.__doc__ = autopara_docstring(op.__doc__, input_iterator_contents)
#         return f
#
# it also requires that the code parsing the stack trace to match up the RemoteInfo dict
# detects the presence of "autoparallelize" in the stack trace, and instead uses
#
#     inspect.getfile(op) + "::" + op.__name__
#
# instead of the stack trace file and function
#
# sphinx docstring parsing cannot handle this, though, because the module associated with
# the "parallelized_op" symbol is that of "autoparallelize", rather than the python file in which
# it is actually defined.  As a result, sphinx assumes it's just some imported symbol and does
# not include its docstring.  There does not appear to be any way to override that on a
# per-symbol basis in current sphinx versions.

_autopara_docstring_params_pre = (
"""inputs: iterable({iterable_type})
    input quantities of type {iterable_type}
outputs: OutputSpec or None
    where to write output atomic configs, or None for no output (i.e. only side-effects)
""")

_autopara_docstring_params_post = (
"""autopara_info: AutoparaInfo, default None
    information for automatic parallelization
""")

autopara_docstring_returns = (
"""Returns
-------
    co: ConfigSet
        output configs
""")

def autoparallelize_docstring(orig_docstring, input_iterable_type):
    output_docstring = ""
    lines = orig_docstring.splitlines(True)
    prev_line_was_parameters_section = False
    for li in range(len(lines)):
        prev_line_match = re.match(r'^(\s*)Parameters\s*[:]?\s*$', lines[li-1])
        if (li >= 1 and re.match(r'^\s*[-]*\s*$', lines[li]) and prev_line_match):
            # set flag so that autopara_docstring_params_pre can be inserted starting on the next line
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
            output_docstring += ''.join([param_init_space + l for l in _autopara_docstring_params_pre.format(iterable_type=input_iterable_type).splitlines(True)])
            # no longer need to insert autoparallelize-related lines
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
    output_docstring += '\n' + ''.join([param_init_space + l for l in _autopara_docstring_params_post.splitlines(True)])

    # insert Returns section
    output_docstring += '\n' + ''.join([section_init_space + l for l in autopara_docstring_returns.splitlines(True)])

    return output_docstring


def autoparallelize(func, *args, def_autopara_info={}, **kwargs):
    """autoparallelize a function

    Use by defining function "op" which takes an input iterable and returns list of configs, and _after_ do

    .. code-block:: python

        def autoparallelized_op(*args, **kwargs):
            return autoparallelize(op, *args, def_autopara_info={"autoparallelize_keyword_param_1": val, "autoparallelize_keyword_param_2": val, ... }, **kwargs )
        autoparallelized_op.doc = autopara_docstring(op.__doc__, "iterable_contents")

    The autoparallelized function can then be called with 

    .. code-block:: python

        parallelized_op(inputs, outputs, [args of op], autopara_info=AutoparaInfo(arg1=val1, ...), [kwargs of op])


    Parameters
    ----------
    func: function
        function to wrap in _autoparallelize_ll()

    *args: list
        positional arguments to func, plus optional first or first and second inputs (iterable) and outputs (OutputSpec) arguments to wrapped function

    def_autopara_info: dict, default {}
        dict with default values for AutoparaInfo constructor keywords setting default autoparallelization info

    **kwargs: dict
        keyword arguments to func, plus optional inputs (iterable), outputs (OutputSpec), and  autopara_info (AutoparaInfo)

    Returns
    -------
    wrapped_func_out: results of calling the function wrapped in autoparallelize via _autoparallelize_ll
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

    autopara_info = kwargs.pop("autopara_info", AutoparaInfo())
    autopara_info.update_defaults(def_autopara_info)

    return _autoparallelize_ll(autopara_info.num_python_subprocesses, autopara_info.num_inputs_per_python_subprocess, inputs, outputs, func,
                               autopara_info.iterable_arg, autopara_info.skip_failed, autopara_info.initializer,
                               autopara_info.remote_info, autopara_info.remote_label, autopara_info.hash_ignore,
                               *args, **kwargs)

# do we want to allow for ops that only take singletons, not iterables, as input, maybe with num_inputs_per_python_subprocess=0?
# that info would have to be passed down to _wrapped_autopara_wrappable so it passes a singleton rather than a list into op
#
# some ifs (int positional vs. str keyword) could be removed if we required that the iterable be passed into a kwarg.

def _autoparallelize_ll(num_python_subprocesses=None, num_inputs_per_python_subprocess=1, iterable=None, outputspec=None,
                        op=None, iterable_arg=0, skip_failed=True, initializer=(None, ()), remote_info=None,
                        remote_label=None, hash_ignore=[], *args, **kwargs):
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
    initializer: (callable, list), default (None, ())
        function to call at beginning of each thread and its positional arguments
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
    remote_info = get_remote_info(remote_info, remote_label)

    if isinstance(iterable_arg, int):
        assert len(args) >= iterable_arg
        # otherwise not enough args were provided

    if outputspec is not None:
        if not isinstance(outputspec, OutputSpec):
            raise RuntimeError('autoparallelize requires outputspec be None or OutputSpec')
        if outputspec.done():
            sys.stderr.write(f'Returning before {op} since output is done\n')
            return outputspec.to_ConfigSet()

    if remote_info is not None:
        out = do_remotely(remote_info, hash_ignore, num_inputs_per_python_subprocess, iterable, outputspec,
                          op, iterable_arg, skip_failed, initializer, args, kwargs)
    else:
        out = do_in_pool(num_python_subprocesses, num_inputs_per_python_subprocess, iterable, outputspec, op, iterable_arg,
                         skip_failed, initializer, args, kwargs)

    return out
