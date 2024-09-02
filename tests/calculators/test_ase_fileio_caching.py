import os

import pytest

from ase.atoms import Atoms


########################
# test Vasp calculator

from tests.calculators.test_vasp import pytestmark as vasp_pytestmark
@vasp_pytestmark
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
# test quantum espresso calculator
from tests.calculators.test_qe import espresso_avail, qe_pseudo
@espresso_avail
def test_qe_cache_timing(tmp_path, monkeypatch, qe_pseudo):
    from ase.calculators.espresso import Espresso as Espresso_ASE
    from wfl.calculators.espresso import Espresso as Espresso_wrap

    config = Atoms('Si', positions=[[0, 0, 9]], cell=[2, 2, 2], pbc=[True, True, True])

    pspot = qe_pseudo
    kwargs_ase =  dict(
        pseudopotentials=dict(Si=pspot.name),
        pseudo_dir=pspot.parent,
        input_data={"SYSTEM": {"ecutwfc": 40, "input_dft": "LDA",}},
        kpts=(2, 3, 4),
        conv_thr=0.0001,
        workdir=tmp_path
    )

    kwargs_wrapper = {}
    cache_timing(config, Espresso_ASE, kwargs_ase, Espresso_wrap, kwargs_wrapper, tmp_path, monkeypatch)


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
