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
# Skip particular tests
# code from Pytest documentation at:
# https://docs.pytest.org/en/latest/example/simple.html#control-skipping-of-tests-according-to-command-line-option
#################################################
wfl_markers = [("slow", "slow tests"),
               ("remote", "tests of remote execution with ExPyRe"),
               ("perf", "tests checking performance")]


def pytest_addoption(parser):
    for marker_name, marker_desc in wfl_markers:
        parser.addoption(f"--run{marker_name}", action="store_true", default=False, help="run " + marker_desc)


def pytest_configure(config):
    for marker_name, marker_desc in wfl_markers:
        config.addinivalue_line("markers", f"{marker_name}: mark {marker_desc}")


def pytest_collection_modifyitems(config, items):
    for marker_name, _ in wfl_markers:
        if not config.getoption(f"--run{marker_name}"):
            skip = pytest.mark.skip(reason=f"need --run{marker_name} option to run")
            for item in items:
                if marker_name in item.keywords:
                    item.add_marker(skip)


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

