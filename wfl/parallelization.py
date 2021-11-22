import os
import re
import shlex
import subprocess
import sys
import time
from inspect import signature
from multiprocessing import Pool

import ase.io
import numpy as np

import wfl.configset


class ParallelDecorator(object):
    """Decorator to make a function work with a parallel Pool

    Use as assignment, not as a @decorator, because pickling of the functions breaks with the latter.

    Parameters
    ----------
    func: callable
        function wrapped

    pool_kwargs: dict
        keyword arguments for the multiprocessing.Pool object

    iterable_argname: str, None
        argument name of iterable to use in func
        None -> first arg

    iterator_method: str, None
        method name to us as iterator
        None -> default, __iter__
    """

    def __init__(self, func, pool_kwargs=None, iterable_argname=None, iterator_method=None):

        if pool_kwargs is None:
            pool_kwargs = {}

        self.func = func  # function to wrap
        self.pool_kwargs = pool_kwargs  # pool parameters
        self.iterable_argname = iterable_argname  # kw for the iterable input
        self.iterator_method = iterator_method  # name of method to use for iteration, None->__iter__

        # todo: add the keywords taken later in the class, like n_pool
        sig = signature(func)
        self.__signature__ = sig.replace()
        self.__doc__ = str(func.__doc__) + "\n ParallelDecorator modification: \n" + self.__doc__

    def __call__(self, *args, **kwargs):
        verbose = "verbose" in kwargs.keys()

        if verbose:
            print("applied this function on iterable")
            print("args:", args)
            print("kwargs", kwargs)

        if "n_pool" in kwargs.keys():
            if self.iterable_argname is None:
                iterable_obj = args[0]
                args = args[1:]
            else:
                iterable_obj = kwargs.pop(self.iterable_argname)

            return self._call_parallel(self._get_iterator(iterable_obj), *args, **kwargs)
        else:
            return self.func(*args, **kwargs)

    def _call_parallel(self, iterable_obj, *args, **kwargs):
        verbose = "verbose" in kwargs.keys()

        # parallel call - the important logic
        pool_kwargs = self.pool_kwargs
        pool_kwargs.update({"processes": kwargs.pop("n_pool")})

        with Pool(**pool_kwargs) as pool:
            result = []
            # put the
            for val in self._get_iterator(iterable_obj):
                if verbose:
                    print("val:", val)
                if self.iterable_argname is not None:
                    kw = kwargs.copy()
                    kw[self.iterable_argname] = val
                    if verbose:
                        print("call in parallel iterable arg:", kw[self.iterable_argname])
                    result.append(pool.apply_async(self.func, args=args, kwds=kw))
                else:
                    result.append(pool.apply_async(self.func, args=[val] + list(args), kwds=kwargs))

            pool.close()
            result = [r.get() for r in result]

        return result

    def _get_iterator(self, iterable_obj):
        # handling custom iterator method of passed inputs, need to predefine it
        if self.iterator_method is None:
            # this will call .__iter__() later on then
            return iterable_obj
        else:
            try:
                return getattr(iterable_obj, self.iterator_method)()
            except AttributeError as e:
                print(f"Object is missing attribute :{self.iterator_method}: to iterate with")
                raise e


def chop_into_files(confset, cost_func, max_cost, output_filename_fmt):
    chopped_input_filenames = []
    current_at_list = []
    chopped_file_i = 0
    current_output_filename = output_filename_fmt.format(chopped_file_i)
    for at_i, at in enumerate(confset):
        if '_chop_at_i' not in at.info:
            at.info['_chop_at_i'] = []
        at.info['_chop_at_i'].append(at_i)
        current_at_list.append(at)
        if cost_func(current_at_list) >= max_cost:
            chopped_input_filenames.append(current_output_filename)
            ase.io.write(current_output_filename, current_at_list)
            current_at_list = []
            chopped_file_i += 1
            current_output_filename = output_filename_fmt.format(chopped_file_i)
    if len(current_at_list) > 0:
        chopped_input_filenames.append(current_output_filename)
        ase.io.write(current_output_filename, current_at_list)
    return chopped_input_filenames


def reaggregate(confset, chopped_output_filenames, do_input=True):
    s_iter = enumerate(confset)
    s_iter_i = None
    if do_input:
        s_iter_i, _ = next(s_iter)
    for fin in chopped_output_filenames:
        print("reaggregate reading from", fin)
        for at in ase.io.read(fin, ':'):
            if do_input:
                if not isinstance(at.info['_chop_at_i'], list):
                    # ASE reads back list of length 1 as scalar
                    at.info['_chop_at_i'] = [at.info['_chop_at_i']]
                while at.info['_chop_at_i'][-1] > s_iter_i:
                    s_iter_i, _ = next(s_iter)
                del at.info['_chop_at_i'][-1]
                if len(at.info['_chop_at_i']) == 0:
                    del at.info['_chop_at_i']
            confset.iwrite(at)
    confset.end_write()
    return confset.get_output()


def str_to_job_cost(func_in):
    if func_in is None or callable(func_in):
        return func_in

    if isinstance(func_in, str):
        if func_in == '_1':
            return lambda at_list: 1
        elif func_in == '_N':
            return lambda at_list: sum([len(at) for at in at_list])
        elif func_in.startswith('_N^') or func_in.startswith('_N**'):
            exponent = float(func_in.replace('_N^', '', 1).replace('_N**', '', 1))
            return lambda at_list: sum([len(at) ** exponent for at in at_list])
        elif isinstance(func_in, str):
            return eval(func_in)
    else:
        raise RuntimeError('Got input function not one of None, callable, or string')


def _do_serial(*args):
    # multiprocessing pool and mpipool seem to handle arguments differently: mult args vs. one tuple with all args
    if len(args) == 1:
        args = args[0]

    if len(args) != 8:
        raise RuntimeError('_do_serial got unknown number of arguments {} != 8'.format(len(args)))

    (serial_command, serial_func, serial_func_kwargs, serial_input_file, serial_output_file, job_specific_args,
     skip_serial_if_complete, serial_output_all_or_none) = args

    if skip_serial_if_complete and os.path.exists(serial_output_file):
        # this is only safe of serial_output_all_or_none is also active
        # calling routine should be checking this
        return

    # NOTE; not ideal that code below assumes and manually re-implements `click` command
    #       line option -> function argument transformation
    if serial_command is not None:
        subprocess_args = serial_command.copy()

        job_specific_args_list = []
        for arg in job_specific_args:
            job_specific_args_list.extend(['{}'.format(arg_comp) for arg_comp in arg])

        subprocess_args += job_specific_args_list + ['--output-file', serial_output_file]

        if serial_input_file is not None:
            subprocess_args.append(serial_input_file)

        sys.stderr.write('doing ' + ' '.join(subprocess_args) + '\n')
        subprocess.run(subprocess_args)
    else:
        use_kwargs = serial_func_kwargs.copy()
        for arg in job_specific_args:
            # do same transformation as click
            arg_name = arg[0].lower()
            arg_name = re.sub('^-+', '', arg[0])
            arg_name = re.sub('-', '_', arg[0])
            if len(arg) == 1:
                use_kwargs.update({arg_name: True})
            elif len(arg) == 2:
                use_kwargs.update({arg_name: arg[1]})
            else:
                use_kwargs.update({arg_name: arg[1:]})
        serial_confset = wfl.configset.ConfigSet(input_files=serial_input_file, output_files=serial_output_file,
                                                 output_all_or_none=serial_output_all_or_none)
        use_kwargs['confset'] = serial_confset
        serial_func(**use_kwargs)


def pre_parallel(multiprocessing_type, npool, serial_command, serial_func, serial_output_all_or_none,
                 skip_serial_if_complete):
    assert multiprocessing_type in ['MPI', 'python']
    if multiprocessing_type == 'python':
        assert isinstance(npool, int)

    if skip_serial_if_complete:
        if not serial_output_all_or_none:
            raise RuntimeError(
                'Got skip_serial_if_complete, but serial_output_all_or_none is not set, refusing to run unsafely')

    if isinstance(serial_command, str):
        serial_command = shlex.split(serial_command)

    assert sum([serial_func is not None, serial_command is not None and len(serial_command) > 0]) == 1

    if multiprocessing_type == 'MPI':
        from mpi4py import MPI
        io_task = (MPI.COMM_WORLD.rank == 0)
    elif multiprocessing_type == 'python':
        MPI = None
        io_task = True
    else:
        raise RuntimeError('Unknown multiprocessing type \'{}\''.format(multiprocessing_type))

    return io_task, MPI


def setup_mp(multiprocessing_type, npool, MPI, bcast_io_data=None):
    if multiprocessing_type == 'MPI':
        MPI.COMM_WORLD.Barrier()

    if multiprocessing_type == 'MPI':
        if npool is not None:
            raise RuntimeError('--multiprocessing MPI does not use --npool')
        from mpipool import Pool
        p = Pool()
        # WARNING: following statement relies on the fact that pre_parallel set io_task=True for rank == 0
        #          should really be cleaned up
        MPI.COMM_WORLD.bcast(bcast_io_data, 0)
    elif multiprocessing_type == 'python':
        if not isinstance(npool, int):
            raise RuntimeError('--multiprocessing python requires --npool INT')
        from multiprocessing.pool import Pool
        p = Pool(npool)
    else:
        raise RuntimeError('Unknown multiprocessing type \'{}\''.format(multiprocessing_type))

    return p


def post_parallel(io_task, confset, serial_input_files, serial_output_files, keep_temp, do_input):
    if io_task:
        s_out = reaggregate(confset, serial_output_files, do_input)
        if not keep_temp:
            if serial_input_files is not None:
                for fin in serial_input_files:
                    os.remove(fin)
            if serial_output_files is not None:
                for fout in serial_output_files:
                    os.remove(fout)
        return s_out
    else:
        return None


def over_N(confset, multiprocessing_type, N, N_arg_name, N_offset_name=None,
           npool=None, max_job_N=1, serial_file_label='serial', keep_temp=False,
           serial_func=None, serial_func_kwargs=None, serial_command=None,
           serial_output_all_or_none=True, skip_serial_if_complete=True):
    """Run multiple instances of a serial task by splitting up some integer input count parameter

    Parameters
    ----------
    confset: ConfigSet
        inputs and outputs of operation

    multiprocessing_type: str, 'MPI' or 'python'
        module to use for spawning parallel processes

    N: int
        number of times to run job

    N_arg_name: str
        name of argument, click-style used to pass number to task

    N_offset_name: str
        name of argument, click-style used to pass offset (when embedding unique sequence number)

    npool: int
        number of processes in pool for multiprocessing_type == multiprocessing
        MPI uses COMM_WORLD, but maybe should optionally use a communicator?

    max_job_N: int, default 1
        maximum value of N for each serial run

    serial_file_label: str, default 'serial'
        label for serial job input/output files

    keep_temp: bool, default False
        keep temporary serial job input/output files

    serial_func: callable, mutually exclusive with serial_command
        function to do in each serial run

    serial_func_kwargs: dict
        kwargs to pass to serial_func

    serial_command: str or list(str)
        command to run with subprocess for each serial task

    serial_output_all_or_none: bool, default True
        pass output_all_or_none to serial task, needed for skip_serial_if_complete to work

    skip_serial_if_complete: bool, default True
        skip serial job if its output file is present

    Returns
    -------
    ConfigSet_out():
        ConfigSet_out for outputs
    """

    if serial_func_kwargs is None:
        serial_func_kwargs = {}
    io_task, MPI = pre_parallel(multiprocessing_type, npool, serial_command, serial_func, serial_output_all_or_none,
                                skip_serial_if_complete)

    p = setup_mp(multiprocessing_type, npool, MPI)

    job_Ns = [max_job_N] * int(N / max_job_N)
    if sum(job_Ns) < N:
        job_Ns.append(N - sum(job_Ns))
    job_N_offsets = [0] + list(np.cumsum(job_Ns)[:-1])

    serial_output_files = ['output.serial.{}.xyz'.format(i) for i in range(len(job_Ns))]
    job_args = []
    for i in range(len(job_Ns)):
        job_args.append((serial_command, serial_func, serial_func_kwargs,
                         None, serial_output_files[i], [('-N', job_Ns[i]), ('--N-offset', job_N_offsets[i])],
                         skip_serial_if_complete, serial_output_all_or_none))
    p.map(_do_serial, job_args)

    return post_parallel(io_task, confset, None, serial_output_files, keep_temp, do_input=False)


def over_inputs(confset, multiprocessing_type,
                npool=None, job_cost_func='_O_1', max_cost=0, serial_file_label='serial', keep_temp=False,
                serial_func=None, serial_func_kwargs=None, serial_command=None,
                serial_output_all_or_none=True, skip_serial_if_complete=True):
    """Run multiple instances of a serial task by splitting up input configs

    Parameters
    ----------
    confset: ConfigSet
        inputs and outputs of operation

    multiprocessing_type: str, 'MPI' or 'python'
        module to use for spawning parallel processes

    npool: int
        number of processes in pool for multiprocessing_type == multiprocessing
        MPI uses COMM_WORLD, but maybe should optionally use a communicator?

    job_cost_func: str or callable
        function giving cost of a serial job for a list(Atoms). Callable needs to be a function
        that takes one argument, the list of atoms. String is '_O_<ORDER>' where ORDER is '1',
        'N', 'N^exponent', or an expression that gives a function taking one atoms_list argument
        after eval() (i.e. a lambda with one argument)

    max_cost: int, default 0
        maximum cost per serial run.  inputs will be added to a single run until total cost >= max.

    serial_file_label: str, default 'serial'
        label for serial job input/output files

    keep_temp: bool, default False
        keep temporary serial job input/output files

    serial_func: callable, mutually exclusive with serial_command
        function to do in each serial run

    serial_func_kwargs: dict
        kwargs to pass to serial_func

    serial_command: str or list(str)
        command to run with subprocess for each serial task

    serial_output_all_or_none: bool, default True
        pass output_all_or_none to serial task, needed for skip_serial_if_complete to work

    skip_serial_if_complete: bool, default True
        skip serial job if its output file is present

    Returns
    -------
    ConfigSet_out():
        ConfigSet_out for outputs
    """

    if serial_func_kwargs is None:
        serial_func_kwargs = {}
    io_task, MPI = pre_parallel(multiprocessing_type, npool, serial_command, serial_func, serial_output_all_or_none,
                                skip_serial_if_complete)

    if io_task:
        serial_input_files = chop_into_files(confset, str_to_job_cost(job_cost_func), max_cost,
                                             output_filename_fmt='{}.{{}}.xyz'.format(serial_file_label))

    p = setup_mp(multiprocessing_type, npool, MPI, bcast_io_data=serial_input_files)

    serial_output_files = ['output.' + f for f in serial_input_files]
    job_args = []
    for i in range(len(serial_input_files)):
        job_args.append((serial_command, serial_func, serial_func_kwargs,
                         serial_input_files[i], serial_output_files[i], [],
                         skip_serial_if_complete, serial_output_all_or_none))

    p.map(_do_serial, job_args)

    return post_parallel(io_task, confset, serial_input_files, serial_output_files, keep_temp, do_input=True)
