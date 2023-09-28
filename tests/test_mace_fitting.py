import yaml
import os, time
from pathlib import Path
import mace, json
from wfl.fit.mace import run_mace_fit


# This part is used to detect the location of 'run_train.py' file. 
mace_dist_dir = list(Path(mace.__file__).parent.parent.glob("mace-*"))[0]
jsonfile = str(Path(mace_dist_dir / "direct_url.json"))
with open(jsonfile) as f:
	d = json.load(f)
script_dir = d["url"].split(":")[-1][2:]
mace_fit_cmd = f"python {script_dir}/scripts/run_train.py"


#print(mace_fit_cmd)
workdir = os.path.join(os.path.dirname(__file__))

def test_mace_fit(request, tmp_path, monkeypatch, mace_fit_cmd=mace_fit_cmd):

	monkeypatch.chdir(tmp_path)
	print("pwd : ", os.getcwd())
	print('getting fitting data from ', request.fspath)

	fit_config_file = os.path.join(os.path.dirname(request.fspath), 'assets', 'B_DFT_data.xyz')
	params_file_path = os.path.join(os.path.dirname(request.fspath), 'assets', 'mace_fit_parameters.yaml')
	mace_params = yaml.safe_load(Path(params_file_path).read_text())
	mace_params["train_file"] = fit_config_file

	t0 = time.time()
	run_mace_fit(mace_params, mace_name = "test", run_dir=".", mace_fit_cmd=mace_fit_cmd)
	t_run = time.time() - t0

	assert os.path.exists(os.path.join(tmp_path, f"test.model"))
	assert os.path.getsize(os.path.join(tmp_path, f"test.model")) > 0

#	assert os.path.exists(os.path.join(tmp_path, run_dir, f"test_swa.model"))
#	assert os.path.getsize(os.path.join(tmp_path, run_dir, f"test_swa.model")) > 0

