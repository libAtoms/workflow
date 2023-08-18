from ase import Atoms
from ase.calculators.singlepoint import SinglePointCalculator
from pytest import approx, raises

from wfl.calculators.utils import save_results


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
    save_results(at, ["energy", "forces", "stress"])
    assert at.calc is not spc_ref
    for key, val in at.calc.results.items():
        assert spc_ref.results[key] == approx(val)
    assert "charges" not in at.calc.results.keys()

    # prefixed
    prefix = "prefix__"
    at, spc_ref = new_calc()
    at.info["energy"] = 100.  # to see if this is removed
    save_results(at, ["energy", "forces", "stress", "charges"], prefix)
    assert at.calc is None
    for k in ["energy", "stress"]:
        assert at.info[f"{prefix}{k}"] == approx(spc_ref.results[k])
    for k in ["forces", "charges"]:
        assert at.arrays[f"{prefix}{k}"] == approx(spc_ref.results[k])
