import numpy as np

from ase import Atoms
from ase.calculators.singlepoint import SinglePointCalculator
from ase.calculators.emt import EMT
from pytest import approx, raises

from wfl.configset import ConfigSet, OutputSpec

from wfl.utils import misc
from wfl.utils.replace_eval_in_strs import replace_eval_in_strs
from wfl.utils.save_calc_results import save_calc_results

from wfl.generate.md import md
from wfl.generate.optimize import optimize


def test_chunks():
    a = range(20)

    for chunk in misc.chunks(a, 5):
        assert len(chunk) == 5


def test_replace_eval_in_strs():
    v0 = 2
    v1 = 1.03
    d = { 'a': 5, 'b': '_EVAL_no_eval' , 'c': [ '_EVAL_ {v0}*2', '_EVAL_ {v1}*3' ] }
    dref = { 'a': 5, 'b': '_EVAL_no_eval' , 'c': [ 4, 3.1 ] }

    drepl = replace_eval_in_strs(d, {'v0': v0, 'v1': v1}, n_float_sig_figs=2)

    assert drepl == dref


def new_calc():
    at = Atoms("H")
    # spc_ref = SinglePointCalculator(at, energy=0., free_energy=0.5, forces=[[0., 2., 4.]],
    spc_ref = SinglePointCalculator(at, energy=0., free_energy=0.0, forces=[[0., 2., 4.]],
                                    stress=[0., 1., 2., 3., 4., 5.],
                                    charges=[0.])
    at.calc = spc_ref

    return at, spc_ref


def test_save_results():
    # simple
    at, spc_ref = new_calc()
    save_calc_results(at, prefix=None, properties=["energy", "forces", "stress"])
    assert at.calc is not spc_ref
    for key, val in at.calc.results.items():
        assert spc_ref.results[key] == approx(val)
    assert "charges" not in at.calc.results.keys()

    # prefixed
    prefix = "prefix__"
    at, spc_ref = new_calc()
    at.info["energy"] = 100.  # to see if this is removed
    save_calc_results(at, prefix=prefix, properties=["energy", "forces", "stress", "charges"])
    assert at.calc is None
    for k in ["energy", "stress"]:
        assert at.info[f"{prefix}{k}"] == approx(spc_ref.results[k])
    for k in ["forces", "charges"]:
        assert at.arrays[f"{prefix}{k}"] == approx(spc_ref.results[k])


def test_save_results_no_old_props():
    ci = ConfigSet([Atoms('Al', positions=[[0, 0, 0]], cell=[3] * 3, pbc=[True] * 3) * 2])
    calc = EMT()

    rng = np.random.default_rng(seed=5)

    ci_md_1 = md(ci, OutputSpec(), calc, steps=2, dt=1.0, temperature=300, temperature_tau=10.0, rng=rng, results_prefix="stored_prop_md_")
    found_n = 0
    for at in ci_md_1:
        for d in (at.info, at.arrays):
            for k in d:
                if k.startswith("stored_prop_"):
                    found_n += 1
                    assert "_md_" in k
    # each config should have energy, stress, forces, energies and free_energy
    assert found_n == 5 * len(list(ci_md_1))

    ci_md_2 = md(ci_md_1, OutputSpec(), calc, steps=2, dt=1.0, temperature=300, temperature_tau=10.0, rng=rng)
    found_n = 0
    for at in ci_md_2:
        for d in (at.info, at.arrays):
            for k in d:
                if k.startswith("last_op__"):
                    assert "_md_" in k
                    found_n += 1
    assert found_n == 5 * len(list(ci_md_2))

    ci_opt = optimize(ci_md_2, OutputSpec(), calc, steps=2)
    found_n = 0
    for at in ci_opt:
        for d in (at.info, at.arrays):
            for k in d:
                if k.startswith("last_op__"):
                    assert "_optimize_" in k
                    found_n += 1
        import sys, ase.io #DEBUG
        ase.io.write(sys.stdout, at, format="extxyz") #DEBUG
    assert found_n == 5 * len(list(ci_opt))


def test_config_type_update():
    ci = ConfigSet([Atoms('Al', positions=[[0, 0, 0]], cell=[3] * 3, pbc=[True] * 3) * 2])
    calc = EMT()

    rng = np.random.default_rng(seed=5)

    ci = md(ci, OutputSpec(), calc, steps=2, dt=1.0, temperature=300, temperature_tau=10.0, rng=rng, results_prefix="stored_prop_md_")
    for at in ci:
        assert at.info["config_type"] == "MD"

    ci = md(ci, OutputSpec(), calc, steps=2, dt=1.0, temperature=300, temperature_tau=10.0, rng=rng)
    for at in ci:
        assert at.info["config_type"] == "MD:MD"

    ci = optimize(ci, OutputSpec(), calc, steps=2)
    for at in ci:
        assert at.info["config_type"].startswith("MD:MD:optimize_")

    ci = optimize(ci, OutputSpec(), calc, steps=2, update_config_type="overwrite")
    for at in ci:
        assert at.info["config_type"].startswith("optimize_")

    ci = md(ci, OutputSpec(), calc, steps=2, dt=1.0, temperature=300, temperature_tau=10.0, rng=rng,
                 update_config_type=False)
    for at in ci:
        assert at.info["config_type"].startswith("optimize_")
