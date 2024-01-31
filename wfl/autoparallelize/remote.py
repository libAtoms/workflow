import sys
import os
import warnings

from ase.atoms import Atoms

from wfl.configset import ConfigSet, OutputSpec
from .utils import items_inputs_generator, set_autopara_per_item_info
from .pool import do_in_pool

from expyre import ExPyRe, ExPyReJobDiedError


def do_remotely(autopara_info, iterable=None, outputspec=None, op=None, rng=None, args=[], kwargs={}, quiet=False, wait_for_results=True):
    """run tasks as series of remote jobs

    Parameters
    ----------
    autopara_info: AutoparaInfo
        object with all information on autoparallelizing remote job, including autopara_info.remote_info
        which contains RemoteInfo with information on remote job, including system, resources, job
        num_inputs_per_python_subprocess, etc, or dict of kwargs for its constructor
    quiet: bool, default False
        do not output (to stderr) progress info

    See autoparallelize.autoparallelize() for other args
    """
    if ExPyRe is None:
        raise RuntimeError('Cannot run as remote jobs since expyre module could not be imported')

    remote_info = autopara_info.remote_info
    autopara_info.remote_info = None

    if autopara_info.num_python_subprocesses is None:
        # number of python processes for autoparallelized remote job not specfied explicitly
        if all([not var.startswith("WFL_NUM_PYTHON_SUBPROCESSES=") for var in remote_info.env_vars]):
            # user didn't explicitly set WFL_NUM_PYTHON_SUBPROCESSES for remote job
            # so set its default equal to number of cores per node
            remote_info.env_vars += ["WFL_NUM_PYTHON_SUBPROCESSES=${EXPYRE_NUM_CORES_PER_NODE}"]

    if remote_info.num_inputs_per_queued_job < 0:
        remote_info.num_inputs_per_queued_job = -remote_info.num_inputs_per_queued_job * autopara_info.num_inputs_per_python_subprocess

    items_inputs = items_inputs_generator(iterable, remote_info.num_inputs_per_queued_job, rng)

    # other per-item info
    inherited_per_item_info = kwargs.get("_autopara_per_item_info")

    # create all jobs (count on expyre detection of identical jobs to avoid rerunning things unnecessarily)
    xprs = []
    # place to keep track of input files, one per input item, so that output can go to corresponding file
    input_locs = []
    # list of all items, wastes space so used only if remote_info.ignore_failed_jobs is True
    all_items = []
    for chunk_i, items_gen in enumerate(items_inputs):
        item_list = []
        item_i_list = []
        rng_list = []
        for (item, item_i, cur_input_loc, rng) in items_gen:
            if isinstance(item, Atoms):
                # special things to do when item is Atoms
                if 'EXPYRE_REMOTE_JOB_FAILED' in item.info:
                    del item.info['EXPYRE_REMOTE_JOB_FAILED']

            item_list.append(item)
            item_i_list.append(item_i)
            input_locs.append(cur_input_loc)
            rng_list.append(rng)

        set_autopara_per_item_info(kwargs, op, inherited_per_item_info, rng_list, item_i_list)

        if remote_info.ignore_failed_jobs:
            all_items.append(item_list)

        job_name = remote_info.job_name + f'_chunk_{chunk_i}'
        if not quiet:
            sys.stderr.write(f'Creating job {job_name}\n')

        if isinstance(iterable, ConfigSet):
            job_iterable = ConfigSet(item_list)
        else:
            job_iterable = item_list
        co = OutputSpec()

        # NOTE: would it be cleaner if remote function was autoparallelize() instead of do_in_pool()
        #
        # ignore OutputSpec for hashing of inputs, since that doesn't affect function
        #   calls that have to happen (also, it's not repeatable for some reason)
        xprs.append(ExPyRe(name=job_name, pre_run_commands=remote_info.pre_cmds, post_run_commands=remote_info.post_cmds,
                            hash_ignore=remote_info.hash_ignore + ['outputspec', 'num_python_subprocesses', 'num_inputs_per_python_subprocess'],
                            env_vars=remote_info.env_vars, input_files=remote_info.input_files,
                            output_files=remote_info.output_files, function=do_in_pool,
                            kwargs={'num_python_subprocesses': autopara_info.num_python_subprocesses,
                                    'num_inputs_per_python_subprocess': autopara_info.num_inputs_per_python_subprocess,
                                    'iterable': job_iterable, 'outputspec': co, 'op': op,
                                    'iterable_arg': autopara_info.iterable_arg,
                                    'skip_failed': autopara_info.skip_failed,
                                    'initializer': autopara_info.initializer,
                                    'args': args, 'kwargs': kwargs}))

    # start jobs (shouldn't do anything if they've already been started)
    for xpr in xprs:
        if not quiet:
            sys.stderr.write(f'Starting job for {xpr.id}\n')
        xpr.start(resources=remote_info.resources, system_name=remote_info.sys_name, header_extra=remote_info.header_extra,
                  exact_fit=remote_info.exact_fit, partial_node=remote_info.partial_node)

    if not wait_for_results:
        return None

    if remote_info.resubmit_killed_jobs:
        # need to loop over all jobs and get results with timeout 0, to look for all failures
        for xpr in xprs:
            try:
                ats_out, stdout, stderr = xpr.get_results(timeout=0, check_interval=0)
            except ExPyReJobDiedError:
                # job has actually failed, resubmit
                warnings.warn(f"Failed job {xpr.id} died, wiping remote and resubmitting")
                xpr.start(resources=remote_info.resources, system_name=remote_info.sys_name, header_extra=remote_info.header_extra,
                          exact_fit=remote_info.exact_fit, partial_node=remote_info.partial_node, force_rerun=True)
            except Exception:
                # ignore all other exceptions here, including ExPyReTimeoutError and real remote exceptions,
                # deal with them below
                pass

    # gather results and write them to original outputspec
    at_i = 0
    for chunk_i, xpr in enumerate(xprs):
        if not quiet:
            sys.stderr.write(f'Gathering results for {xpr.id} remote {xpr.remote_id}\n')

        stdout = None
        stderr = None
        try:
            ats_out, stdout, stderr = xpr.get_results(timeout=remote_info.timeout, check_interval=remote_info.check_interval)
        except Exception:
            print("stdout", "-" * 30)
            print(stdout)
            print("stderr", "-" * 30)
            print(stderr)
            warnings.warn(f'Failed in remote job {xpr.id} on {xpr.system_name}')
            if not remote_info.ignore_failed_jobs:
                raise
            if len(all_items) > 0 and isinstance(all_items[chunk_i][0], Atoms):
                # get ready to write input configs to output
                ats_out = ConfigSet(all_items[chunk_i])
                for at in ats_out:
                    at.info['EXPYRE_REMOTE_JOB_FAILED'] = True
            else:
                # either no inputs saved or inputs aren't configurations, so skip output
                ats_out = None
            stdout = ''
            stderr = ''

        if ats_out is None:
            # Skip the right number of input files. If we're here,
            # remote_info.ignore_failed_jobs must be True, so all_items should be filled
            at_i += len(all_items[chunk_i])
        else:
            for at in ats_out.groups():
                outputspec.store(at, input_locs[at_i])
                at_i += 1
            sys.stdout.write(stdout)
            sys.stderr.write(stderr)

    outputspec.close()

    if 'WFL_EXPYRE_NO_MARK_PROCESSED' not in os.environ:
        # mark as processed only after outputspec has been finished
        for xpr in xprs:
            xpr.mark_processed()

    return outputspec.to_ConfigSet()
