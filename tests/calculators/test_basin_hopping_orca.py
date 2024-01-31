from os import path
from shutil import rmtree

import numpy as np
from ase import Atoms
from ase.build import molecule
from ase.calculators.calculator import CalculationFailed
from pytest import approx, raises

from wfl.calculators.orca.basinhopping import BasinHoppingORCA


def test_orca_utils(tmp_path):
    # setup
    at_ch4 = molecule("CH4")
    at_ch3 = molecule("CH3")

    calc_even = BasinHoppingORCA(scratchdir=tmp_path, rng=np.random.default_rng(1))
    calc_even.atoms = at_ch4
    calc_odd = BasinHoppingORCA(rng=np.random.default_rng(1))
    calc_odd.atoms = at_ch3

    # HOMO
    assert calc_even.get_homo() == (4, 4)
    assert calc_odd.get_homo() == (4, 3)

    # multiplicity
    assert calc_even.get_multiplicity() == 1
    assert calc_odd.get_multiplicity() == 2

    # new atoms
    new_at_even: Atoms = calc_even._copy_atoms()
    assert new_at_even.positions == approx(at_ch4.positions)
    assert new_at_even.get_atomic_numbers() == approx(at_ch4.get_atomic_numbers())

    # scratch paths
    generated_scratch_dir = calc_even._make_tempdir()
    assert f"{tmp_path}/orca_" in generated_scratch_dir

    if path.isdir(generated_scratch_dir):
        rmtree(generated_scratch_dir)


def test_orca_process_results(tmp_path):
    # setup
    at_ch4 = molecule("CH4")
    calc = BasinHoppingORCA(scratchdir=tmp_path, forces_tol=0.05, rng=np.random.default_rng(1))
    calc.atoms = at_ch4

    # shape errors, correct are (3, 10) and (3, 10, 5, 3)
    with raises(AssertionError):
        _ = calc.process_results(np.zeros(shape=(2, 10)), np.zeros(shape=(3, 10, 5, 3)))
        _ = calc.process_results(np.zeros(shape=(3, 10)), np.zeros(shape=(3, 100, 5, 3)))

    # none succeeding
    e = np.zeros(shape=(3, 10)) + np.inf
    f = np.zeros(shape=(3, 10, 5, 3))
    with raises(CalculationFailed, match="Not enough runs succeeded.*0.*in wavefunction basin hopping"):
        _ = calc.process_results(e, f)

    # one of three
    e[0, 1] = 0.999
    with raises(CalculationFailed, match="Not enough runs succeeded.*1.*in wavefunction basin hopping"):
        _ = calc.process_results(e, f)

    # 2 of three passes
    e[1, 2] = 1.0001
    _ = calc.process_results(e, f)

    # energy difference
    e[2, 4] = 0.0
    with raises(CalculationFailed, match="Too high energy difference found: *1.0.*"):
        _ = calc.process_results(e, f)

    # force error not triggered on non-minimum frame's force diff
    e[2, 4] = np.inf
    f[0, -1] = 100
    _ = calc.process_results(e, f)

    # error on forces
    f[1, 2, 0, 0] = 0.051
    with raises(CalculationFailed, match="Too high force difference found: *0.051.*"):
        _ = calc.process_results(e, f)

    # results tested
    f[0, 1] = 0.025
    e_out, force_out = calc.process_results(e, f)
    assert e_out == 0.999
    assert force_out == approx(0.025)
