import os
import sys
import subprocess
import shutil
import warnings
import tempfile

import ase.io

from expyre import ExPyRe
from wfl.configset import ConfigSet
from wfl.autoparallelize.utils import get_remote_info
from pathlib import Path
from shutil import copyfile


def fit(fitting_configs, mace_name, mace_fit_params, mace_fit_cmd=None, ref_property_prefix="REF_",
        prev_checkpoint_file=None, valid_configs=None, test_configs=None, skip_if_present=True, run_dir=".",
        verbose=True, dry_run=False, remote_info=None, remote_label=None, wait_for_results=True):
    """
        Fit MACE model.


    **Environment Variables**

    * WFL_MACE_FIT_COMMAND: command to execute mace fit, e.g.
      ``python $HOME/mace/scripts/run_train.py ``


    Parameters
    ----------
    fitting_configs: ConfigSet
       set of configurations to fit (mace param "train_file")
    mace_name: str
        name of MACE label
    mace_fit_params: dict
        parameters for fitting a MACE model.
    mace_fit_cmd: str, default None.
        command for excecuting the MACE fitting. (For example, "python ~/path_to_mace_cripts/run_train.py")
        Alternatively set by `WFL_MACE_FIT_COMMAND` env var.
    ref_property_prefix: str, default "REF_"
        string prefix added to atoms.info/arrays keys (energy, forces, virial, stress)
    prev_checkpoint_file: str, default None
        Previous checkpoint file to restart from.
    valid_configs: ConfigSet, default None
        set of configurations to validate (mace param "valid_file")
    test_configs: ConfigSet, default None
        set of configurtions to test (mace param "test_file")
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

    # fill in some params from standard function arguments
    mace_fit_params["name"] = mace_name
    mace_fit_params["energy_key"] = ref_property_prefix + "energy"
    mace_fit_params["forces_key"] = ref_property_prefix + "forces"
    if "compute_stress" in mace_fit_params:
        mace_fit_params["stress_key"] = ref_property_prefix + "stress"

    assert isinstance(mace_fit_params, dict)
    if prev_checkpoint_file is not None:
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
        # run dir will contain only things created by fitting, so it's safe to copy the
        # entire thing back as output
        output_files = remote_info.output_files + [str(run_dir)]

        # convert to lists in memory so pickling for remote run will work
        fitting_configs = ConfigSet(list(fitting_configs))
        if valid_configs is not None:
            valid_configs = ConfigSet(list(valid_configs))
        if test_configs is not None:
            test_configs = ConfigSet(list(test_configs))

        # set number of threads in queued job, only if user hasn't set them
        if not any([var.split('=')[0] == 'WFL_MACE_FIT_OMP_NUM_THREADS' for var in remote_info.env_vars]):
            remote_info.env_vars.append('WFL_MACE_FIT_OMP_NUM_THREADS=$EXPYRE_NUM_CORES_PER_NODE')
        if not any([var.split('=')[0] == 'WFL_NUM_PYTHON_SUBPROCESSES' for var in remote_info.env_vars]):
            remote_info.env_vars.append('WFL_NUM_PYTHON_SUBPROCESSES=$EXPYRE_NUM_CORES_PER_NODE')

        remote_func_kwargs = {'fitting_configs': fitting_configs, 'mace_name': mace_name,
                              'mace_fit_params': mace_fit_params, 'remote_info': '_IGNORE', 'run_dir': run_dir,
                              'mace_fit_cmd': mace_fit_cmd, 'prev_checkpoint_file': prev_checkpoint_file,
                              'valid_configs': valid_configs, 'test_configs': test_configs}

        xpr = ExPyRe(name=remote_info.job_name, pre_run_commands=remote_info.pre_cmds, post_run_commands=remote_info.post_cmds,
                     env_vars=remote_info.env_vars, input_files=input_files, output_files=output_files, function=fit,
                     kwargs=remote_func_kwargs)

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

    if valid_configs is not None:
        valid_configs_scratch_filename = _prep_configs_file(valid_configs, mace_fit_params, "valid_file")
    if test_configs is not None:
        test_configs_scratch_filename = _prep_configs_file(test_configs, mace_fit_params, "test_file")

    if mace_fit_cmd is None:
        if os.environ.get("WFL_MACE_FIT_COMMAND") is not None:
            mace_fit_cmd = os.environ.get("WFL_MACE_FIT_COMMAND")
        elif shutil.which("mace_run_train") is not None:
            mace_fit_cmd = shutil.which("mace_run_train")
        else:
            raise Exception("Path for run_train.py not found.")

    fitting_configs_scratch_filename = _prep_configs_file(fitting_configs, mace_fit_params, "train_file")

    if mace_fit_params.get("foundation_model", None) is not None and Path(mace_fit_params["foundation_model"]).is_file():
        mace_fit_params["foundation_model"] = str(Path(mace_fit_params["foundation_model"]).absolute())

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
        if dry_run:
            warnings.warn("Exiting mace.fit without fitting, because dry_run is True")
            return None

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
        if valid_configs is not None and valid_configs_scratch_filename is not None:
            Path(valid_configs_scratch_filename).unlink()
        if test_configs is not None and test_configs_scratch_filename is not None:
            Path(test_configs_scratch_filename).unlink()

    except subprocess.CalledProcessError as exc:
        print("Failure in calling MACE fitting with error code:", exc.returncode)
        raise exc

    if orig_omp_n is not None:
        os.environ["OMP_NUM_THREADS"] = orig_omp_n


def _prep_configs_file(configs, use_params, key):
    """
    Writes configs to file and updates MACE fitting parameters.
    Configurations and filename handled by Workflow overwrite any filename
    specified in parameters.

    Parameters:
    -----------
    configs: ConfigSet
        configurations to write to a file (fitting, validation, testing, etc)
    use_params: dict
        MACE fit parameters, will have input filename set based on where configs were written to

    Return:
    -------
    filename
        temporary file name or None if already file is written beforehand
    """

    configs_filename = configs.one_file()

    if not configs_filename:
        fd_scratch, filename = tempfile.mkstemp(prefix=f"_MACE_{key}_configs.", suffix=".xyz", dir=".")
        os.close(fd_scratch)

        if key in use_params.keys():
            warnings.warn(f"Ignoring configs file '{use_params[key]}' in mace_fit_params, "
                          f"instead using configs passed in and saved to '{filename}'.")

        use_params[key] = filename
        ase.io.write(filename, configs)

        return filename

    else:
        use_params[key] = configs_filename

        return None
