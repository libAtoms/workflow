import yaml, os, shutil
import time
from pathlib import Path
import mace, json
from wfl.fit.mace import fit
from wfl.configset import ConfigSet
import ase.io
import pytest

if not os.environ.get("WFL_MACE_FIT_COMMAND") and shutil.which("mace_run_train") is None:
    pytestmark = pytest.mark.skip(reason="No mace_run_train found in WFL_MACE_FIT_COMMAND or path")

def test_mace_fit_from_list(request, tmp_path, monkeypatch):

    monkeypatch.chdir(tmp_path)
    print('getting fitting data from ', request.path)
    print("here :  ", request.path)
	
    parent_path = request.path.parent
    params_file_path = parent_path / 'assets' / 'mace_fit_parameters.yaml'
    mace_fit_params = yaml.safe_load(params_file_path.read_text())
    filepath = parent_path / 'assets' / 'B_DFT_data_mace_ftting.xyz'
    fitting_configs = ConfigSet(ase.io.read(filepath, ":"))

    t0 = time.time()
    fit(fitting_configs, "test", mace_fit_params, mace_fit_cmd=None, run_dir=".")
    t_run = time.time() - t0

    assert (tmp_path / "test.model").exists()
    assert (tmp_path / "test.model").stat().st_size > 0


def test_mace_fit(request, tmp_path, monkeypatch):

    monkeypatch.chdir(tmp_path)
    print('getting fitting data from ', request.path)
    print("here :  ", request.path)

    parent_path = request.path.parent
    fit_config_file = parent_path / 'assets' / 'B_DFT_data_mace_ftting.xyz'
    params_file_path = parent_path / 'assets' / 'mace_fit_parameters.yaml'
    mace_fit_params = yaml.safe_load(params_file_path.read_text())
    fitting_configs = ConfigSet(fit_config_file) 

    t0 = time.time()
    fit(fitting_configs, "test", mace_fit_params, mace_fit_cmd=None, run_dir=".")
    t_run = time.time() - t0

    assert (tmp_path / "test.model").exists()
    assert (tmp_path / "test.model").stat().st_size > 0


@pytest.mark.remote
def test_mace_fit_from_list_remote(request, tmp_path, monkeypatch, expyre_systems, remoteinfo_env):
    run_remote_test(test_mace_fit_from_list, request, tmp_path, monkeypatch, expyre_systems, remoteinfo_env)


@pytest.mark.remote
def test_mace_fit_remote(request, tmp_path, monkeypatch, expyre_systems, remoteinfo_env):
    run_remote_test(test_mace_fit, request, tmp_path, monkeypatch, expyre_systems, remoteinfo_env)


def run_remote_test(test_func, request, tmp_path, monkeypatch, expyre_systems, remoteinfo_env):
    env_vars = ['WFL_ACE_FIT_OMP_NUM_THREADS=$EXPYRE_NUM_CORES_PER_NODE']
    # add things that often have to be customized for MACE fitting to work. Should 
    # this kind of mechanism be more generalized rather than hard wired here?
    # after all, the person who knows the environment in which the tests are run 
    # should know what env vars will need to be replicated in the remote job
    if "PYTHONPATH" in os.environ:
        env_vars += ['PYTHONPATH="' + os.environ["PYTHONPATH"] + '"']
    if "LD_PRELOAD" in os.environ:
        env_vars += ['LD_PRELOAD="' + os.environ["LD_PRELOAD"] + '"']

    if "WFL_MACE_FIT_COMMAND" in os.environ:
        env_vars += ["WFL_MACE_FIT_COMMAND=" + os.environ["WFL_MACE_FIT_COMMAND"]]
    ri = {'resources' : {'max_time': '10m', 'num_nodes': 1},
          'pre_cmds': [ f'export PYTHONPATH={Path(__file__).parent.parent}:$PYTHONPATH'],
          'env_vars' : env_vars}

    for sys_name in expyre_systems:
        if sys_name.startswith('_'):
            continue

        ri['sys_name'] = sys_name
        ri['job_name'] = 'pytest_mace_fit_'+sys_name

        remoteinfo_env(ri)

        monkeypatch.setenv('WFL_EXPYRE_INFO', json.dumps(ri))
        test_func(request, tmp_path, monkeypatch) # , run_dir=f'run_dir_{sys_name}')
