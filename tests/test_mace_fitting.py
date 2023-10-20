import yaml, os, shutil
import time
from pathlib import Path
import mace, json
from wfl.fit.mace import fit
from wfl.configset import ConfigSet
import ase.io
import pytest

os.environ["WFL_MACE_FIT_COMMAND"] = "python /raven/u/hjung/Softwares/mace/scripts/run_train.py"
#print("hi: ", os.environ.get("WFL_MACE_FIT_COMMAND"))

@pytest.mark.skipif(not os.environ.get("WFL_MACE_FIT_COMMAND") and shutil.which("mace_run_train") is None, reason="No mace_run_train found in WFL_MACE_FIT_COMMAND or path")
def test_mace_fit_from_list(request, tmp_path, monkeypatch):

    monkeypatch.chdir(tmp_path)
    print('getting fitting data from ', request.path)
    print("here :  ", request.path)
	
    parent_path = request.path.parent
    params_file_path = parent_path / 'assets' / 'mace_fit_parameters.yaml'
    mace_fit_params = yaml.safe_load(params_file_path.read_text())
    filepath = parent_path / 'assets' / 'B_DFT_data.xyz'
    fitting_configs = ConfigSet(ase.io.read(filepath, ":"))

    t0 = time.time()
    fit(fitting_configs, "test", mace_fit_params, mace_fit_cmd=None, run_dir=".")
    t_run = time.time() - t0

    assert (tmp_path / "test.model").exists()
    assert (tmp_path / "test.model").stat().st_size > 0


#@pytest.mark.skipif(not os.environ.get("WFL_MACE_FIT_COMMAND"), shutil.which("mace_run_train") is None, reason="No mace_run_train found in WFL_MACE_FIT_COMMAND or path")
#def test_mace_fit(request, tmp_path, monkeypatch):
#
#    monkeypatch.chdir(tmp_path)
#    print('getting fitting data from ', request.path)
#    print("here :  ", request.path)
#
#    parent_path = request.path.parent
#    fit_config_file = parent_path / 'assets' / 'B_DFT_data.xyz'
#    params_file_path = parent_path / 'assets' / 'mace_fit_parameters.yaml'
#    mace_fit_params = yaml.safe_load(params_file_path.read_text())
#    fitting_configs = ConfigSet(fit_config_file) 
#
#    t0 = time.time()
#    fit(fitting_configs, "test", mace_fit_params, mace_fit_cmd=None, run_dir=".")
#    t_run = time.time() - t0
#
#    assert (tmp_path / "test.model").exists()
#    assert (tmp_path / "test.model").stat().st_size > 0


