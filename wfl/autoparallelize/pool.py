import sys
import os
import warnings
import inspect

import functools

# seems to conflict with torch's own multiprocessing
# try:
#     # make multiprocessing use dill instead of pickle, mainly to support lambdas.  See
#     #   https://stackoverflow.com/questions/19984152/what-can-multiprocessing-and-dill-do-together
#     import dill, multiprocessing
#     dill.Pickler.dumps, dill.Pickler.loads = dill.dumps, dill.loads
#     multiprocessing.reduction.ForkingPickler = dill.Pickler
#     multiprocessing.reduction.dump = dill.dump
# except ModuleNotFoundError:
#     pass

from multiprocessing.pool import Pool

from wfl.configset import ConfigSet
from wfl.autoparallelize.mpipool_support import wfl_mpipool

from .utils import grouper, items_inputs_generator, set_autopara_per_item_info


def _wrapped_autopara_wrappable(op, iterable_arg, inherited_per_item_info, args, kwargs, item_inputs):
    """Wrap an operation to be run in parallel by autoparallelize

    Parameters:
    -----------
        op: callable
            function to call
        iterable_arg: int/str
            where to put iterable item argument. If int, place in positional args,
                or if str, key in kwargs
        inherited_per_item_info: list(dict)
            list of per-item info
        args: list
            list of positional args
        kwargs: dict
            dict of keyword args
        item_inputs: iterable(4-tuples)
            One or more 4-tuples. (item_i, item, label, item_rng). item is passed to function in iterable_arg,
            item_i is its number in the overall list, label is a quantity to be passed back with the
            output, and rng is a numpy.random.Generator to use for that item

    Returns:
    -------
        list of 2-tuples containing, the output of each function call, together with the corresponding label (3rd) field of item_inputs
    """
    for item_input in item_inputs:
        assert len(item_input) == 4

    # item_inputs is iterable of 4-tuples
    item_list = [item_input[0] for item_input in item_inputs]
    item_i_list = [item_input[1] for item_input in item_inputs]
    label_list = [item_input[2] for item_input in item_inputs]
    rng_list = [item_input[3] for item_input in item_inputs]

    set_autopara_per_item_info(kwargs, op, inherited_per_item_info, rng_list, item_i_list)

    if isinstance(iterable_arg, int):
        u_args = args[0:iterable_arg] + (item_list,) + args[iterable_arg:]
    else:
        u_args = args
        if iterable_arg is not None:
            kwargs[iterable_arg] = item_list

    outputs = op(*u_args, **kwargs)
    if outputs is None:
        outputs = [None] * len(item_list)
    return zip(outputs, label_list)


# do we want to allow for ops that only take singletons, not iterables, as input, maybe with num_inputs_per_python_subprocess=0?
# that info would have to be passed down to _wrapped_autopara_wrappable so it passes a singleton rather than a list into op
#
# some ifs (int positional vs. str keyword) could be removed if we required that the iterable be passed into a kwarg.
def do_in_pool(num_python_subprocesses=None, num_inputs_per_python_subprocess=1, iterable=None, outputspec=None, op=None, iterable_arg=0,
               skip_failed=True, initializer=(None, []), rng=None, args=[], kwargs={}):
    """parallelize some operation over an iterable

    Parameters
    ----------
    num_python_subprocesses: int, default os.environ['WFL_NUM_PYTHON_SUBPROCESSES']
        number of processes to parallelize over, 0 for running in serial
    num_inputs_per_python_subprocess: int, default 1
        number of items from iterable to pass to kach invocation of operation
    iterable: iterable, default None
        iterable to loop over, often ConfigSet but could also be other things like range()
    outputspec: OutputSpec, defaulat None
        object containing returned Atoms objects
    op: callable
        function to call with each chunk
    iterable_arg: itr or str, default 0
        positional argument or keyword argument to place iterable items in when calling op
    skip_failed: bool, default True
        skip function calls that return None
    initializer: (callable, list), default (None, [])
        function to call at beginning of each thread and its positional args
    args: list
        positional arguments to op
    kwargs: dict
        keyword arguments to op

    Returns
    -------
    ConfigSet containing returned configs if outputspec is not None, otherwise None
    """
    assert len(initializer) == 2, f"Bad initializer {initializer}"

    if num_python_subprocesses is None:
        num_python_subprocesses = int(os.environ.get('WFL_NUM_PYTHON_SUBPROCESSES', 0))

    # actually do the work locally
    did_no_work = True

    items_inputs = items_inputs_generator(iterable, num_inputs_per_python_subprocess, rng)

    # other per-item info
    inherited_per_item_info = kwargs.get("_autopara_per_item_info")

    if num_python_subprocesses > 0:
        # use multiprocessing
        op_full_name = inspect.getmodule(op).__name__ + "." + op.__name__
        sys.stderr.write(f'Running {op_full_name} with num_python_subprocesses={num_python_subprocesses}, '
                         f'num_inputs_per_python_subprocess={num_inputs_per_python_subprocess}\n')
        if wfl_mpipool:
            # MPI pool is global and unique, created at script start, so do not create one here
            warnings.warn(f'mpipool ignores > 0 value of num_python_subprocesses={num_python_subprocesses}, '
                          f'always uses all MPI processes {wfl_mpipool.size}')
            if initializer[0] is not None:
                # generate a task for each mpi process that will call initializer with positional initargs
                _ = wfl_mpipool.map(functools.partial(_wrapped_autopara_wrappable, initializer[0], None, None, initializer[1], {}),
                                    grouper(1, ((None, None, None, None) for i in range(wfl_mpipool.size))))
            pool = wfl_mpipool
        else:
            if initializer[0] is not None:
                initializer_args = {'initializer': initializer[0], 'initargs': initializer[1]}
            else:
                initializer_args = {}
            pool = Pool(num_python_subprocesses, **initializer_args)

        if wfl_mpipool:
            map_f = pool.map
        else:
            map_f = pool.imap
        results = map_f(functools.partial(_wrapped_autopara_wrappable, op, iterable_arg,
                                          inherited_per_item_info, args, kwargs), items_inputs)

        if not wfl_mpipool:
            # only close pool if it's from multiprocessing.pool
            pool.close()

        # always loop over results to trigger lazy imap()
        for result_group in results:
            if outputspec is not None:
                for at, from_input_loc in result_group:
                    if skip_failed and at is None:
                        continue
                    did_no_work = False
                    outputspec.store(at, from_input_loc)

        if not wfl_mpipool:
            # call join pool (prevent pytest-cov deadlock as per https://pytest-cov.readthedocs.io/en/latest/subprocess-support.html)
            # but only if it's from multiprocessing.pool
            pool.join()

    else:
        # do directly: still not trivial because of num_inputs_per_python_subprocess
        # NOTE: this does not pickle configs to send to the remote processes, so called function
        # can change configs in-place, which is different from Pool.map().  Should we pickle and
        # unpickle to better reproduce the behavior of Pool.map() ?
        for items_inputs_group in items_inputs:
            result_group = _wrapped_autopara_wrappable(op, iterable_arg, inherited_per_item_info, args,
                                                       kwargs, items_inputs_group)

            if outputspec is not None:
                for at, from_input_loc in result_group:
                    if skip_failed and at is None:
                        continue
                    did_no_work = False
                    outputspec.store(at, from_input_loc)

    if outputspec is not None:
        outputspec.close()
        if did_no_work:
            return ConfigSet(None)
        else:
            return outputspec.to_ConfigSet()
    else:
        return None
