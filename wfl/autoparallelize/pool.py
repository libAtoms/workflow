import sys
import os
import warnings

import functools
from multiprocessing.pool import Pool

from wfl.configset import ConfigSet
from wfl.autoparallelize.mpipool_support import wfl_mpipool

from .utils import grouper


def _wrapped_autopara_wrappable(op, iterable_arg, args, kwargs, item_inputs):
    """Wrap an operation to be run in parallel by pipeline

    Parameters:
    -----------
        op: callable
            function to call
        iterable_arg: int/str
            where to put iterable item argument. If int, place in positional args,
                or if str, key in kwargs
        args: list
            list of positional args
        kwargs: dict
            dict of keyword args
        item_inputs: iterable(2-tuples)
            One or more 2-tuples. First field of each is quantities to be passed to function in
                iterable_arg, and second field is a quantity to be passed back with the output.

    Returns:
    -------
        list of 2-tuples containing, for each first field of 2-tuples in item_inputs, the output of each
            function call, together with the corresponding 2nd field of item_inputs
    """
    for item in item_inputs:
        assert len(item) == 2

    # item_inputs is iterable of 2-tuples
    item_list = [item_input[0] for item_input in item_inputs]

    if isinstance(iterable_arg, int):
        u_args = args[0:iterable_arg] + (item_list,) + args[iterable_arg:]
    else:
        u_args = args
        if iterable_arg is not None:
            kwargs[iterable_arg] = item_list

    outputs = op(*u_args, **kwargs)
    if outputs is None:
        outputs = [None] * len(item_list)
    return zip(outputs, [item_input[1] for item_input in item_inputs])


# do we want to allow for ops that only take singletons, not iterables, as input, maybe with chunksize=0?
# that info would have to be passed down to _wrapped_autopara_wrappable so it passes a singleton rather than a list into op
#
# some ifs (int positional vs. str keyword) could be removed if we required that the iterable be passed into a kwarg.
def do_in_pool(npool=None, chunksize=1, iterable=None, configset_out=None, op=None, iterable_arg=0,
               skip_failed=True, initializer=None, initargs=None, args=[], kwargs={}):
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

    if npool is None:
        npool = int(os.environ.get('WFL_AUTOPARA_NPOOL', 0))

    # actually do the work locally
    if configset_out is not None:
        configset_out.pre_write()

    did_no_work = True

    if isinstance(iterable, ConfigSet):
        items_inputs_generator = grouper(chunksize, ((item, iterable.get_current_input_file()) for item in iterable))
    else:
        items_inputs_generator = grouper(chunksize, ((item, None) for item in iterable))

    if npool > 0:
        # use multiprocessing
        sys.stderr.write(f'Running {op} with npool={npool}, chunksize={chunksize}\n')
        if wfl_mpipool:
            # MPI pool is global and unique, created at script start, so do not create one here
            warnings.warn(f'mpipool ignores > 0 value of npool={npool}, '
                          f'always uses all MPI processes {wfl_mpipool.size}')
            if initializer is not None:
                # generate a task for each mpi process that will call initializer with positional initargs
                _ = wfl_mpipool.map(functools.partial(_wrapped_autopara_wrappable, initializer, None, initargs, {}),
                                    grouper(1, ((None, None) for i in range(wfl_mpipool.size))))
            pool = wfl_mpipool
        else:
            if initializer is not None:
                initializer_args = {'initializer': initializer, 'initargs': initargs}
            else:
                initializer_args = {}
            pool = Pool(npool, **initializer_args)

        if wfl_mpipool:
            map_f = pool.map
        else:
            map_f = pool.imap
        results = map_f(functools.partial(_wrapped_autopara_wrappable, op, iterable_arg, args, kwargs), items_inputs_generator)

        if not wfl_mpipool:
            # only close pool if its from multiprocessing.pool
            pool.close()

        # always loop over results to trigger lazy imap()
        for result_group in results:
            if configset_out is not None:
                for at, from_input_file in result_group:
                    if skip_failed and at is None:
                        continue
                    did_no_work = False
                    configset_out.write(at, from_input_file=from_input_file)

    else:
        # do directly, still not trivial because of chunksize
        for items_inputs_group in items_inputs_generator:
            result_group = _wrapped_autopara_wrappable(op, iterable_arg, args, kwargs, items_inputs_group)

            if configset_out is not None:
                for at, from_input_file in result_group:
                    if skip_failed and at is None:
                        continue
                    did_no_work = False
                    configset_out.write(at, from_input_file=from_input_file)

    if configset_out is not None:
        configset_out.end_write()
        if did_no_work:
            return ConfigSet()
        else:
            return configset_out.to_ConfigSet_in()
    else:
        return None
