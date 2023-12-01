import json
import os
import time

from pathlib import Path

import pytest

from wfl.configset import ConfigSet
from wfl.fit.utils import ace_fit_jl_path
from wfl.fit.ace import fit, prepare_params, prepare_configs

try:
    ace_fit_jl_path()
except:
    pytestmark = pytest.mark.skip(reason="ace_fit_jl_path failed to find ace fitting script")

def test_ace_fit_dry_run(request, tmp_path, monkeypatch, run_dir='run_dir'):
    print('getting fitting data from ', request.fspath)

    # kinda ugly, but remote running of multistage fit doesn't support absolute run_dir, so test
    # with a relative one
    monkeypatch.chdir(tmp_path)
    (tmp_path / run_dir).mkdir()

    fit_config_file = os.path.join(os.path.dirname(request.fspath), 'assets', 'B_DFT_data.xyz')

    # Only mandatory params
    params = {
        "basis": {
            "ace": {"type": "ace", "species": ["B"], "N": 3, "maxdeg": 6},
            "pair": {"type": "pair", "species": ["B"], "maxdeg": 4}},
        "solver": {"type": "lsqr"}}

    t0 = time.time()
    ACE_size = fit(
        fitting_configs=ConfigSet(fit_config_file),
        ACE_name="ACE.B_test",
        ace_fit_params=params, 
        ref_property_prefix="REF_",
        run_dir=str(run_dir), 
        dry_run=True, 
        skip_if_present=True)
    time_actual = time.time() - t0

    assert len(ACE_size) == 2
    assert isinstance(ACE_size[0], int) and isinstance(ACE_size[1], int)

    assert os.path.exists(os.path.join(tmp_path, run_dir, f'ACE.B_test.size'))

    t0 = time.time()
    ACE_size_rerun = fit(
        fitting_configs=ConfigSet(fit_config_file),
        ACE_name="ACE.B_test",
        ace_fit_params=params, 
        ref_property_prefix="REF_",
        run_dir=str(run_dir), 
        dry_run=True, 
        skip_if_present=True)
    time_rerun = time.time() - t0

    assert ACE_size == ACE_size_rerun

    # rerun should reuse files, be much faster
    assert time_rerun < time_actual / 10

def test_ace_fit(request, tmp_path, monkeypatch, run_dir='run_dir'):
    print('getting fitting data from ', request.fspath)

    # kinda ugly, but remote running of multistage fit doesn't support absolute run_dir, so test
    # with a relative one
    monkeypatch.chdir(tmp_path)
    (tmp_path / run_dir).mkdir()

    fit_config_file = os.path.join(os.path.dirname(request.fspath), 'assets', 'B_DFT_data.xyz')

    # Only mandatory params
    params = {
        "data": {},
        "basis": {
            "ace": {"type": "ace", "species": ["B"], "N": 3, "maxdeg": 6},
            "pair": {"type": "pair", "species": ["B"], "maxdeg": 4}},
        "solver": {"type": "lsqr"}}


    t0 = time.time()
    ACE = fit(
        fitting_configs=ConfigSet(fit_config_file),
        ACE_name="ACE.B_test",
        ace_fit_params=params, 
        ref_property_prefix="REF_",
        run_dir=str(run_dir), 
        skip_if_present=True)
    time_actual = time.time() - t0
    print('ACE', ACE)

    assert os.path.exists(os.path.join(tmp_path, run_dir, f'ACE.B_test.json'))

    print(params)

    t0 = time.time()
    ACE = fit(
        fitting_configs=ConfigSet(fit_config_file),
        ACE_name="ACE.B_test",
        ace_fit_params=params, 
        ref_property_prefix="REF_",
        run_dir=str(run_dir), 
        skip_if_present=True)
    time_rerun = time.time() - t0

    # rerun should reuse files, be much faster
    assert time_rerun < time_actual / 10

def test_ace_fit_basis_size(request, tmp_path, monkeypatch, run_dir='run_dir'):
    print('getting fitting data from ', request.fspath)

    # kinda ugly, but remote running of multistage fit doesn't support absolute run_dir, so test
    # with a relative one
    monkeypatch.chdir(tmp_path)
    (tmp_path / run_dir).mkdir()

    fit_config_file = os.path.join(os.path.dirname(request.fspath), 'assets', 'B_DFT_data.xyz')

    # Simple basis
    params = {
        "basis": {
            "ace": {"type": "ace", "species": ["B"], "N": 3, "maxdeg": 6},
            "pair": {"type": "pair", "species": ["B"], "maxdeg": 4}},
        "solver": {"type": "lsqr"}}

    ACE_size = fit(
        fitting_configs=ConfigSet(fit_config_file),
        ACE_name="ACE.B_test",
        ace_fit_params=params, 
        ref_property_prefix="REF_",
        run_dir=str(run_dir), 
        dry_run=True, 
        skip_if_present=False)

    assert ACE_size == [1047, 29]

    # different degree for one correlation order, int dict key
    params = {
        "basis": {
            "ace": {"type": "ace",
                    "species": ["B"],
                    "N": 3,
                    "maxdeg": 1.0,
                    "degree": { "type": "sparseM",
                                "Dd": { "default": 6,
                                        3: 4
                                      }
                               }
                    },
            "pair": {"type": "pair", "species": ["B"], "maxdeg": 4}},
        "solver": {"type": "lsqr"}}

    ACE_size = fit(
        fitting_configs=ConfigSet(fit_config_file),
        ACE_name="ACE.B_test",
        ace_fit_params=params, 
        ref_property_prefix="REF_",
        run_dir=str(run_dir), 
        dry_run=True, 
        skip_if_present=False)

    assert ACE_size == [1047, 23]

    # different degree for one species and correlation order, tuple dict key
    params = {
        "basis": {
            "ace": {"type": "ace",
                    "species": ["B"],
                    "N": 3,
                    "maxdeg": 1.0,
                    "degree": { "type": "sparseM",
                                "Dd": { "default": 6,
                                        (3, "B"): 4
                                      }
                               }
                    },
            "pair": {"type": "pair", "species": ["B"], "maxdeg": 4}},
        "solver": {"type": "lsqr"}}

    ACE_size = fit(
        fitting_configs=ConfigSet(fit_config_file),
        ACE_name="ACE.B_test",
        ace_fit_params=params, 
        ref_property_prefix="REF_",
        run_dir=str(run_dir), 
        dry_run=True, 
        skip_if_present=False)

    assert ACE_size == [1047, 23]


@pytest.mark.remote
def test_ace_fit_remote(request, tmp_path, expyre_systems, monkeypatch, remoteinfo_env):
    env_vars = ['WFL_ACE_FIT_JULIA_THREADS=$( [ $EXPYRE_NUM_CORES_PER_NODE -gt 2 ] && echo 2 || echo $(( $EXPYRE_NUM_CORES_PER_NODE )) )',
                'WFL_ACE_FIT_BLAS_THREADS=$EXPYRE_NUM_CORES_PER_NODE']
    if "WFL_ACE_FIT_COMMAND" in os.environ:
        env_vars += ["WFL_ACE_FIT_COMMAND=" + os.environ["WFL_ACE_FIT_COMMAND"]]
    ri = {'resources' : {'max_time': '10m', 'num_nodes': 1},
          'pre_cmds': [ f'export PYTHONPATH={Path(__file__).parent.parent}:$PYTHONPATH'],
          'env_vars' : env_vars}

    for sys_name in expyre_systems:
        if sys_name.startswith('_'):
            continue

        ri['sys_name'] = sys_name
        ri['job_name'] = 'pytest_ace_fit_'+sys_name

        remoteinfo_env(ri)

        monkeypatch.setenv('WFL_EXPYRE_INFO', json.dumps(ri))
        test_ace_fit(request, tmp_path, monkeypatch, run_dir=f'run_dir_{sys_name}')
