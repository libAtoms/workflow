import os, sys, yaml, subprocess
from expyre import ExPyRe
from wfl.configset import ConfigSet, OutputSpec
from wfl.autoparallelize.utils import get_remote_info
from expyre.resources import Resources
from pathlib import Path 
from shutil import copyfile



def run_mace_fit(params, mace_fit_cmd, mace_name="mace", run_dir=".", remote_info=None,
		verbose=True, do_fit=True, wait_for_results=True, remote_label=None, skip_if_present=True, **kwargs):
	"""
		Fit MACE model.
	Parameters
	----------
	params: str or dict
		parameters for fitting the model, it can be directly read from YAML file or passed on as dict. 
	mace_fit_cmd: str
		command for excecuting the MACE fitting. (For example, "python ~/path_to_mace_cripts/run_train.py")
	mace_name: str
	 	name of MACE label
	run_dir: str, default '.'
		directory to run fitting in
	remote_info: dict or wfl.autoparallelize.utils.RemoteInfo, or '_IGNORE' or None
        If present and not None and not '_IGNORE', RemoteInfo or dict with kwargs for RemoteInfo
        constructor which triggers running job in separately queued job on remote machine.  If None,
        will try to use env var WFL_EXPYRE_INFO used (see below). '_IGNORE' is for
        internal use, to ensure that remotely running job does not itself attempt to spawn another
        remotely running job.
	verbose: bool default True
		verbose output
	do_fit: bool, default True
		carry out the fit, otherwise only print fitting command 
	wait_for_results: bool, default True
		wait for results of remotely executed job, otherwise return after starting job
	remote_label: str, default None
		label to match in WFL_EXPYRE_INFO
	skip_if_present: bool, default False
		skip if final GAP file exists in expected place
	
	"""
	run_dir = Path(run_dir)

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


		# set number of threads in queued job, only if user hasn't set them
		if not any([var.split('=')[0] == 'WFL_GAP_FIT_OMP_NUM_THREADS' for var in remote_info.env_vars]):
			remote_info.env_vars.append('WFL_GAP_FIT_OMP_NUM_THREADS=$EXPYRE_NUM_CORES_PER_NODE')
		if not any([var.split('=')[0] == 'WFL_NUM_PYTHON_SUBPROCESSES' for var in remote_info.env_vars]):
			remote_info.env_vars.append('WFL_NUM_PYTHON_SUBPROCESSES=$EXPYRE_NUM_CORES_PER_NODE')

		remote_func_kwargs = {'params': params,'remote_info': '_IGNORE', 'run_dir': run_dir,
							'input_files' : remote_info.input_files.copy(), 
							"mace_fit_cmd" : mace_fit_cmd}

		kwargs.update(remote_func_kwargs)
		xpr = ExPyRe(name=remote_info.job_name, pre_run_commands=remote_info.pre_cmds, post_run_commands=remote_info.post_cmds,
					 env_vars=remote_info.env_vars, input_files=input_files, output_files=output_files, function=run_mace_fit,
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

	if not run_dir.exists():
		run_dir.mkdir(parents=True)

	if isinstance(params, str):
		params = yaml.safe_load(Path(params).read_text())
	elif isinstance(params, dict):
		pass

	for key, val in params.items():
		if isinstance(val, int) or isinstance(val, float):
			mace_fit_cmd += f" --{key}={val}"
		elif isinstance(val, str):
			mace_fit_cmd += f" --{key}='{val}'"
		elif val is None:
			mace_fit_cmd += f" --{key}"
		else:
			mace_fit_cmd += f" --{key}='{val}'"
	
	if not do_fit or verbose:
		print('fitting command:\n', mace_fit_cmd)


	orig_omp_n = os.environ.get('OMP_NUM_THREADS', None)
	if 'WFL_GAP_FIT_OMP_NUM_THREADS' in os.environ:
		os.environ['OMP_NUM_THREADS'] = os.environ['WFL_GAP_FIT_OMP_NUM_THREADS']
	
	try:
		remote_cwd = os.getcwd()	
		if str(run_dir) != ".":
			for input_file in kwargs["input_files"]:
				file_name = input_file.split("/")[-1]
				
				# If initiated with previous checkpoint file, it should be copied to 
				# current fitted directory after creating checkpoint folder. 
				if file_name.endswith("_swa.pt"):
					checkpoint_dir = Path(os.path.join(run_dir, 'checkpoints'))	
					checkpoint_dir.mkdir(parents=True, exist_ok=True)
					copyfile(input_file, f"{checkpoint_dir}/{file_name}")

				else:
					copyfile(input_file, f"{run_dir}/{file_name}")

			os.chdir(run_dir)
			subprocess.run(mace_fit_cmd, shell=True, check=True)
			os.chdir(remote_cwd)	
		else:
			subprocess.run(mace_fit_cmd, shell=True, check=True)

	except subprocess.CalledProcessError as e:
		print("Failure in calling MACE fitting with error code:", e.returncode)
		raise e
	

if __name__ == "__main__":
	cwd = os.getcwd()
	run_mace_fit(f"{cwd}/params.yaml", remote_info = remote_info)


