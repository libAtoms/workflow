import os

import pytest

from ase.atoms import Atoms


########################
# test Vasp calculator

from tests.calculators.test_vasp import test_vasp_mark
@pytest.mark.skipif(test_vasp_mark, reason='Vasp testing env vars missing')
def test_vasp_cache_timing(tmp_path, monkeypatch):
    from ase.calculators.vasp import Vasp as Vasp_ase
    from wfl.calculators.vasp import Vasp as Vasp_wrap

    config = Atoms('Si', positions=[[0, 0, 9]], cell=[2, 2, 2], pbc=[True, True, True])
    kwargs_ase = {'encut': 200, 'pp': os.environ['PYTEST_VASP_POTCAR_DIR']}
    kwargs_wrapper = {'workdir': tmp_path}
    # make sure 'pp' is relative to correct dir (see wfl.calculators.vasp)
    if os.environ['PYTEST_VASP_POTCAR_DIR'].startswith('/'):
        monkeypatch.setenv("VASP_PP_PATH", "/.")
    else:
        monkeypatch.setenv("VASP_PP_PATH", ".")
    cache_timing(config, Vasp_ase, kwargs_ase, Vasp_wrap, kwargs_wrapper, tmp_path, monkeypatch)


########################
# generic code used by all calculators

import time

from wfl.configset import ConfigSet, OutputSpec
from wfl.calculators import generic

def cache_timing(config, calc_ase, kwargs_ase, calc_wfl, kwargs_wrapper, rundir, monkeypatch):
    (rundir / "run_calc_ase").mkdir()

    calc = calc_ase(**kwargs_ase)
    config.calc = calc

    monkeypatch.chdir(rundir / "run_calc_ase")
    t0 = time.time()
    E = config.get_potential_energy()
    ase_time = time.time() - t0

    monkeypatch.chdir(rundir)
    t0 = time.time()
    _ = generic.calculate(inputs=ConfigSet(config), outputs=OutputSpec(),
                          calculator=calc_wfl(**kwargs_wrapper, **kwargs_ase))
    wfl_time = time.time() - t0

    print("ASE", ase_time, "WFL", wfl_time)

    assert wfl_time < ase_time * 1.25
