import sys
import os
import re

import pytest

from pathlib import Path
import shutil


def do_init_mpipool():
    import wfl.mpipool_support

    wfl.mpipool_support.init()


@pytest.mark.skipif(
    "WFL_AUTOPARA_MPIPOOL" not in os.environ, reason="only init mpipool if WFL_AUTOPARA_MPIPOOL is in env"
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


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: mark test as slow to run")
    config.addinivalue_line("markers", "remote: mark test as related to remote execution")


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


