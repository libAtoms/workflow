import sys
import os
import re
import json

import pytest

from pathlib import Path
import shutil


@pytest.fixture(autouse=True)
def env_setup(monkeypatch):
    # actually run in parallel by default
    if "WFL_NUM_PYTHON_SUBPROCESSES" not in os.environ:
        monkeypatch.setenv("WFL_NUM_PYTHON_SUBPROCESSES", "2")


def do_init_mpipool():
    import wfl.autoparallelize.mpipool_support

    wfl.autoparallelize.mpipool_support.init()


@pytest.mark.skipif(
    "WFL_MPIPOOL" not in os.environ, reason="only init mpipool if WFL_MPIPOOL is in env"
)
@pytest.mark.mpi(minsize=2)
@pytest.fixture(scope="session", autouse=True)
def init_mpipool(request):
    """initialize mpipool, only if running with mpirun
    """
    do_init_mpipool()

    # request.addfinalizer(finalizer_function)


@pytest.fixture(scope="session")
def quippy():
    return pytest.importorskip("quippy")


################################################
# Skip of slow or remote execution tests
# code from Pytest documentation at:
# https://docs.pytest.org/en/latest/example/simple.html#control-skipping-of-tests-according-to-command-line-option
#################################################
def pytest_addoption(parser):
    parser.addoption(
        "--runslow", action="store_true", default=False, help="run slow tests")
    parser.addoption(
        "--runremote", action="store_true", default=False, help="run remote execution tests"
    )
    parser.addoption(
        "--runperf", action="store_true", default=False, help="run performance tests")


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: mark test as slow to run")
    config.addinivalue_line("markers", "remote: mark test as related to remote execution")
    config.addinivalue_line("markers", "perf: mark test as testing performance")


def pytest_collection_modifyitems(config, items):
    if not config.getoption("--runslow"):
        skip_slow = pytest.mark.skip(reason="need --runslow option to run")
        for item in items:
            if "slow" in item.keywords:
                item.add_marker(skip_slow)
    if not config.getoption("--runremote"):
        skip_remote = pytest.mark.skip(reason="need --runremote option to run")
        for item in items:
            if "remote" in item.keywords:
                item.add_marker(skip_remote)
    if not config.getoption("--runperf"):
        skip_perf = pytest.mark.skip(reason="need --runperf option to run")
        for item in items:
            if "perf" in item.keywords:
                item.add_marker(skip_perf)


### fixture for workflow tests that use expyre, namely tests/test_remote_run.py
@pytest.fixture()
def expyre_systems(tmp_path):
    if not str(tmp_path).startswith(str(Path.home())):
        pytest.xfail(reason='expyre tests require tmp_path be under $HOME, pass "--basetemp $HOME/pytest"')

    expyre_mod = pytest.importorskip('expyre')

    expyre_root = Path(str(tmp_path / '.expyre'))
    expyre_root.mkdir()
    shutil.copy(Path.home() / '.expyre' / 'config.json', expyre_root / 'config.json')

    expyre_mod.config.init(expyre_root, verbose=True)

    try:
        for sys_name in list(expyre_mod.config.systems.keys()):
            if not re.search(os.environ['EXPYRE_PYTEST_SYSTEMS'], sys_name):
                sys.stderr.write(f'Not using {sys_name}, does not match regexp in EXPYRE_PYTEST_SYSTEMS\n')
                del expyre_mod.config.systems[sys_name]
    except KeyError:
        pass

    return expyre_mod.config.systems

@pytest.fixture()
def remoteinfo_env():
    def remoteinfo_env_func(ri):
        if 'WFL_PYTEST_EXPYRE_INFO' in os.environ:
            ri_extra = json.loads(os.environ['WFL_PYTEST_EXPYRE_INFO'])
            if 'resources' in ri_extra:
                ri['resources'].update(ri_extra['resources'])
                del ri_extra['resources']
            ri.update(ri_extra)

        # add current wfl directory to PYTHONPATH early, so it's used for remote jobs
        if 'env_vars' not in ri:
            ri['env_vars'] = []
        ri['env_vars'].append(f'PYTHONPATH={Path(__file__).parent.parent}:$PYTHONPATH')

    return remoteinfo_env_func

