import os, sys, yaml, subprocess
import warnings, tempfile
import ase.io
from expyre import ExPyRe
from wfl.configset import ConfigSet, OutputSpec
from wfl.autoparallelize.utils import get_remote_info
from expyre.resources import Resources
from pathlib import Path 
from shutil import copyfile


def fit(fitting_configs, mace_name, mace_fit_params, mace_fit_cmd, ref_property_prefix="REF_", 
        prev_checkpoint_file=None, skip_if_present=True, run_dir=".", verbose=True, dry_run=False, 
        remote_info=None, remote_label=None, wait_for_results=True):
    """
        Fit MACE model.

    Parameters
    ----------
    fitting_configs: ConfigSet
       set of configurations to fit 
    mace_name: str
        name of MACE label
    mace_fit_params: dict
        parameters for fitting a MACE model.
    mace_fit_cmd: str
        command for excecuting the MACE fitting. (For example, "python ~/path_to_mace_cripts/run_train.py")
    ref_property_prefix: str, default "REF\_"
        string prefix added to atoms.info/arrays keys (energy, forces, virial, stress)
    prev_checkpoint_file: str, default None
        Previous checkpoint file to restart from. 
    skip_if_present: bool, default True
        skip if final MACE file exists in expected place
    run_dir: str, default '.'
        directory to run fitting in
    verbose: bool default True
        verbose output
    dry_run: bool, default False
        do a dry run, and returns fitting command including keywords
    remote_info: dict or wfl.autoparallelize.utils.RemoteInfo, or '_IGNORE' or None
        If present and not None and not '_IGNORE', RemoteInfo or dict with kwargs for RemoteInfo
        constructor which triggers running job in separately queued job on remote machine.  If None,
        will try to use env var WFL_EXPYRE_INFO used (see below). '_IGNORE' is for
        internal use, to ensure that remotely running job does not itself attempt to spawn another
        remotely running job.
    remote_label: str, default None
        label to match in WFL_EXPYRE_INFO
    wait_for_results: bool, default True
        wait for results of remotely executed job, otherwise return after starting job
    """ 
    run_dir = Path(run_dir)
  
    assert isinstance(mace_fit_params, dict)
    if prev_checkpoint_file != None: 
        assert Path(prev_checkpoint_file).is_file(), "No previous checkpoint file found!"
        
    if skip_if_present:
        try:
            print(f"check whether already fitted model exists as {run_dir}/{mace_name}.model")
            if not Path(f"{run_dir}/{mace_name}.model").is_file():
                raise FileNotFoundError
    
            return mace_name
        except (FileNotFoundError, RuntimeError):
            pass
    
    if remote_info != '_IGNORE':
        remote_info = get_remote_info(remote_info, remote_label)
    
    if remote_info is not None and remote_info != '_IGNORE':
        input_files = remote_info.input_files.copy()
        output_files = remote_info.output_files.copy() + [str(run_dir)]
   
#        fitting_configs = ConfigSet(list(fitting_configs))

        # set number of threads in queued job, only if user hasn't set them
        if not any([var.split('=')[0] == 'WFL_MACE_FIT_OMP_NUM_THREADS' for var in remote_info.env_vars]):
            remote_info.env_vars.append('WFL_MACE_FIT_OMP_NUM_THREADS=$EXPYRE_NUM_CORES_PER_NODE')
        if not any([var.split('=')[0] == 'WFL_NUM_PYTHON_SUBPROCESSES' for var in remote_info.env_vars]):
            remote_info.env_vars.append('WFL_NUM_PYTHON_SUBPROCESSES=$EXPYRE_NUM_CORES_PER_NODE')
    
        remote_func_kwargs = {'fitting_configs': fitting_configs, 'mace_name': mace_name,
                            'mace_fit_params': mace_fit_params, 'remote_info': '_IGNORE', 'run_dir': run_dir,
                            "mace_fit_cmd" : mace_fit_cmd, 'prev_checkpoint_file' : prev_checkpoint_file}
    
        xpr = ExPyRe(name=remote_info.job_name, pre_run_commands=remote_info.pre_cmds, post_run_commands=remote_info.post_cmds,
                     env_vars=remote_info.env_vars, input_files=input_files, output_files=output_files, function=fit,
                     kwargs = remote_func_kwargs)
    
        xpr.start(resources=remote_info.resources, system_name=remote_info.sys_name, header_extra=remote_info.header_extra,
                  exact_fit=remote_info.exact_fit, partial_node=remote_info.partial_node)
    
        if not wait_for_results:
            return None
        results, stdout, stderr = xpr.get_results(timeout=remote_info.timeout, check_interval=remote_info.check_interval)
    
        sys.stdout.write(stdout)
        sys.stderr.write(stderr)
    
        # no outputs to rename since everything should be in run_dir
        xpr.mark_processed()
        
        return results
    
    run_dir.mkdir(parents=True, exist_ok=True)
   
    fitting_configs_scratch_filename = _prep_fitting_configs_file(fitting_configs, mace_fit_params)

    for key, val in mace_fit_params.items():
        if isinstance(val, int) or isinstance(val, float):
            mace_fit_cmd += f" --{key}={val}"
        elif isinstance(val, str):
            mace_fit_cmd += f" --{key}='{val}'"
        elif val is None:
            mace_fit_cmd += f" --{key}"
        else:
            mace_fit_cmd += f" --{key}='{val}'"
 
    if dry_run or verbose:
        print('fitting command:\n', mace_fit_cmd)
    
    orig_omp_n = os.environ.get('OMP_NUM_THREADS', None)
    if 'WFL_MACE_FIT_OMP_NUM_THREADS' in os.environ:
        os.environ['OMP_NUM_THREADS'] = os.environ['WFL_MACE_FIT_OMP_NUM_THREADS']
    
    try:
        remote_cwd = os.getcwd()    

        # previous checkpoint file should be moved by remote_info.input_files function 
        if prev_checkpoint_file is not None:
            checkpoint_dir = run_dir / "checkpoints"
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            # check if file exists in the destination. 
            try:
                copyfile(prev_checkpoint_file, f"{checkpoint_dir}/{Path(prev_checkpoint_file).stem}.pt")
            except shutil.SameFileError:
                pass

        os.chdir(run_dir)
        subprocess.run(mace_fit_cmd, shell=True, check=True)
        os.chdir(remote_cwd)    
        
        if fitting_configs_scratch_filename is not None:
            Path(fitting_configs_scratch_filename).unlink()

    except subprocess.CalledProcessError as e:
        print("Failure in calling MACE fitting with error code:", e.returncode)
        raise e


def _prep_fitting_configs_file(fitting_configs, use_params):
    """
    Writes fitting configs to file and updates MACE fitting parameters.
    Configurations and filename handled by Workflow overwrite any filename
    specified in parameters.

    Parameters:
    -----------
    fitting_configs: ConfigSet
        configurations to fit to
    use_params: dict
        MACE fit parameters, will have input filename set based on where configs were written to

    Return:
	-------
    filename
        temporary file name or None if already file is written beforehand 
    """

    fitting_configs_filename = fitting_configs.one_file()

    if not fitting_configs_filename:
        fd_scratch, filename = tempfile.mkstemp(prefix="_MACE_fitting_configs.", suffix=".xyz")    
        os.close(fd_scratch)

        if "train_file" in use_params.keys():
            warnings.warn(f"Ignoring configs file '{use_params['train_file']}' in mace_fit_params, "
                          f"instead using configs passed in and saved to '{filename}'.")
    
        use_params["train_file"] = filename
        ase.io.write(filename, fitting_configs)

        return filename

    else:
        use_params["train_file"] = fitting_configs_filename 

        return None

