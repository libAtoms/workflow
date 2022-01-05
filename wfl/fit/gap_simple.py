import sys
import os
import subprocess
import json
from pathlib import Path
from copy import deepcopy

from wfl.configset import ConfigSet_in
from wfl.utils.quip_cli_strings import dict_to_quip_str
from .utils import to_RemoteInfo
from .gap_relocate import gap_relocate

from expyre import ExPyRe

def run_gap_fit(fitting_configs, fitting_dict, stdout_file, gap_fit_exec="gap_fit",
                verbose=True, do_fit=True, remote_info=None, **kwargs):
    """Runs gap_fit

    Parameters
    ----------
    fitting_configs: str, ConfigSet_in
        filename or set of configurations to fit
    fitting_dict: dict
        dict of keys to turn into command line for gap_fit
    stdout_file: str
        filename to pass standard output to
    gap_fit_exec: str, default "gap_fit"
        executable for gap_fit
    verbose: bool, default True
    do_fit: bool, default True
        carry out the fit, otherwise only print fitting command
    remote_info: dict or wfl.pipeline.utils.RemoteInfo, or '_IGNORE' or None
        If present and not None and not 'IGNORE_NONE', RemoteInfo or dict with kwargs for RemoteInfo
        constructor which triggers running job in separately queued job on remote machine.  If None,
        will try to use env var WFL_GAP_SIMPLE_FIT_REMOTEINFO used (see below). '_IGNORE' is for
        internal use, to ensure that remotely running job does not itself attempt to spawn another
        remotely running job.
    kwargs
        any key:val pair will be added to the fitting str

    Environment Variables
    ---------------------
    WFL_GAP_SIMPLE_FIT_REMOTEINFO: JSON dict or name of file containing JSON with kwargs for RemoteInfo
        contructor to be used to run fitting in separate queued job
    GAP_FIT_OMP_NUM_THREADS: number of threads to set for OpenMP of gap_fit
    """

    remote_info = to_RemoteInfo(remote_info, 'WFL_GAP_SIMPLE_FIT_REMOTEINFO')
    assert 'atoms_filename' not in fitting_dict and 'at_file' not in fitting_dict

    if isinstance(fitting_configs, str):
        fitting_configs = ConfigSet_in(input_files=fitting_configs)

    if remote_info is not None and remote_info != '_IGNORE':
        input_files = remote_info.input_files.copy()
        output_files = remote_info.output_files.copy()

        # the code below needs to know an unfortunate amount about the inner workings of gap_fit

        # put configs in memory so they can be staged out easily
        fitting_configs = fitting_configs.in_memory()

        # here we rely on knowledge of the default gap_file and the correpsonding output files

        # Make remote fit write the GAP xml file in the current directory, so we don't have
        # to create any directory hierarchy on the remote machine.  If gap_file _is_ in a subdirectory
        # this may overwrite a file with the same name in the current directory when it's staged back
        # Therefore, we rename it by adding an '_', and save the new name in use_gap_file
        gap_file = fitting_dict.get('gap_file', 'GAP.xml')
        use_gap_file = '_' + Path(gap_file).name
        fitting_dict['gap_file'] = use_gap_file
        if use_gap_file + '*' not in output_files:
            output_files.append(use_gap_file + '*')

        # add stdout file, again renamed so as not to overwrite
        use_stdout_file = '_' + Path(stdout_file).name
        if use_stdout_file not in output_files:
            output_files.append(use_stdout_file)

        # set number of threads in queued job, if user didn't already request a specific number
        if all([not env_var == 'GAP_FIT_OMP_NUM_THREADS' and
                not env_var.startswith('GAP_FIT_OMP_NUM_THREADS=') for env_var in remote_info.env_vars]):
            remote_info.env_vars.append('GAP_FIT_OMP_NUM_THREADS=$EXPYRE_NCORES_PER_NODE')

        remote_func_kwargs = {'fitting_configs': fitting_configs, 'fitting_dict': fitting_dict,
                              'stdout_file': use_stdout_file,
                              'gap_fit_exec': gap_fit_exec, 'verbose': verbose, 'do_fit': do_fit,
                              'remote_info': '_IGNORE'}
        remote_func_kwargs.update(kwargs)
        xpr = ExPyRe(name=remote_info.job_name, pre_run_commands=remote_info.pre_cmds, post_run_commands=remote_info.post_cmds,
                      env_vars=remote_info.env_vars, input_files=input_files, output_files=output_files, function=run_gap_fit,
                      kwargs=remote_func_kwargs)

        xpr.start(resources=remote_info.resources, system_name=remote_info.sys_name,
                  exact_fit=remote_info.exact_fit, partial_node=remote_info.partial_node)
        results, stdout, stderr = xpr.get_results()
        sys.stdout.write(stdout)
        sys.stderr.write(stderr)

        # move gap_file (and everything related) to desired path
        gap_relocate(use_gap_file, gap_file, delete_old=True)
        Path(use_stdout_file).rename(stdout_file)

        # now that all files have been saved, mark as processed so it can be cleaned later
        xpr.mark_processed()

        return results

    # convert fitting configs to a single file
    fitting_configs_scratch_filename = None
    fitting_configs_filename = fitting_configs.is_one_file()
    if not fitting_configs_filename:
        fitting_configs_scratch_filename = fitting_configs.to_scratch_file('_GAP_fitting_configs.xyz')
        fitting_configs_filename = fitting_configs_scratch_filename
    
    # kwargs overwrite the fitting_dict given
    use_fitting_dict = dict(fitting_dict, atoms_filename=fitting_configs_filename, **kwargs)

    fitting_line = dict_to_gap_fit_string(use_fitting_dict)

    cmd = f'{gap_fit_exec} {fitting_line} 2>&1 > {stdout_file} '

    if not do_fit or verbose:
        print('fitting command:\n', cmd)

    if not do_fit:
        return

    orig_omp_n = os.environ.get('OMP_NUM_THREADS', None)
    if 'GAP_FIT_OMP_NUM_THREADS' in os.environ:
        os.environ['OMP_NUM_THREADS'] = os.environ['GAP_FIT_OMP_NUM_THREADS']

    # this will raise an error if return status is not 0
    # we could also capture stdout and stderr here, but right now that's done by shell
    try:
        subprocess.run(cmd, shell=True, check=True)
    except subprocess.CalledProcessError as e:
        print("Failure in calling GAP fitting with error code:", e.returncode)
        raise e

    # run can fail without raising an exception in subprocess.run, at least make sure that
    # GAP file exists
    assert Path(fitting_dict.get('gap_file', 'GAP.xml')).exists()

    if fitting_configs_scratch_filename is not None:
        Path(fitting_configs_scratch_filename).unlink()

    if orig_omp_n is not None:
        os.environ['OMP_NUM_THREADS'] = str(orig_omp_n)


def dict_to_gap_fit_string(param_dict):
    """ converts dictionary with gap_fit parameters to string for calling
    gap_fit.

    - booleans -> "T"/"F"
    - lists ->  { v1 v2 v3 ... }
    - {'key1':[v1, v2, v3], 'key2':[v1, v2, v3]}
        -> '{key1:v1:v2:v3:key2:v1:v2:v3}'
    - strings with spaces get enclosed in quotes
    - otherwise {key:val} -> key=val
    - asserts that mandatory parameters are given
    - descriptors are passed in pram_dict['_gap'], which is a list of
      dictionaries, one dictionary per descriptor

    """

    param_dict = deepcopy(param_dict)

    assert 'atoms_filename' in param_dict.keys() or 'at_file' in param_dict.keys()
    assert '_gap' in param_dict.keys()
    assert len(param_dict['_gap']) > 0
    assert 'default_kernel_regularisation' in param_dict.keys() or 'default_sigma' in param_dict.keys()

    descriptors = param_dict.pop('_gap')

    param_dict = _Path_to_str(param_dict)
    gap_fit_string = dict_to_quip_str(param_dict)

    descriptor_string_list = [dict_to_quip_str(desc_dict, list_brackets='{{}}') for
                              desc_dict in descriptors]

    gap_fit_string += ' gap={ ' + ' : '.join(descriptor_string_list) + ' }'

    return gap_fit_string

def _Path_to_str(param_dict):

    possible_file_keys = ["atoms_filename", "at_file", "baseline_param_filename", 
                          "core_param_file", "gap_file", "gp_file", "template_file"]

    for key in possible_file_keys:
        if key in param_dict.keys():
            param_dict[key] = str(param_dict[key])

    return param_dict
