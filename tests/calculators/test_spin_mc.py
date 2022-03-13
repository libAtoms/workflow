import numpy as np
from ase.build import bulk, molecule
from pytest import approx, fixture, raises
from wfl.calculators.castep_spin_monte_carlo import (
    CastepSpinMonteCarlo,
    UniformSpinInitializer,
)


@fixture
def sample_atoms():
    # three atoms objects with properties

    at0 = bulk("SiC", crystalstructure="zincblende", a=4.36, cubic=True)
    at0.info["dummy_energy"] = -10
    at0.info["dummy_stress"] = np.arange(6)
    at0.arrays["dummy_forces"] = np.arange(24).reshape(8, 3)

    at1 = bulk("SiC", crystalstructure="zincblende", a=4.36, cubic=True)
    at1.info["dummy_energy"] = -8
    at1.info["dummy_stress"] = np.arange(6) * 0.1
    at1.arrays["dummy_forces"] = np.arange(24).reshape(8, 3) * -0.5

    at2 = bulk("SiC", crystalstructure="zincblende", a=4.36, cubic=True)
    at2.info["dummy_energy"] = -3
    at2.info["dummy_stress"] = np.arange(6) * -0.6
    at2.arrays["dummy_forces"] = np.arange(24).reshape(8, 3) + 1

    return [at0, at1, at2]


def test_uniform_spin_initializer():
    obj = UniformSpinInitializer(10, 20)
    at = molecule("CH4")

    def _check(new_at):
        assert at == new_at
        assert at is not new_at

        spins = new_at.get_initial_magnetic_moments()
        assert np.all(10 < spins)
        assert np.all(spins < 20)

    # sample a single one
    at_single = obj.sample(at)
    _check(at_single)

    # sample multiple
    at_multi = obj.sample_multiple(at, 3)
    assert len(at_multi) == 3
    for at_x in at_multi:
        _check(at_x)


def test_spin_mc_min(sample_atoms):
    obj = CastepSpinMonteCarlo(
        atoms=bulk("SiC", crystalstructure="zincblende", a=4.36, cubic=True),
        method="min",
        output_prefix="dummy_",
    )

    result = obj.choose_result(sample_atoms)
    # this is actually returning the minimum one, unchanged
    assert result is sample_atoms[0]


def test_spin_mc_mean(sample_atoms):
    obj = CastepSpinMonteCarlo(
        atoms=bulk("SiC", crystalstructure="zincblende", a=4.36, cubic=True),
        method="mean",
        output_prefix="dummy_",
    )

    result = obj.choose_result(sample_atoms)
    # this has a copy with changed values
    assert result == sample_atoms[0]
    assert result is not sample_atoms[0]
    assert result.info["dummy_energy"] == approx(-7.0)
    assert result.info["dummy_stress"] == approx(
        [0.0, 0.16666667, 0.33333333, 0.5, 0.66666667, 0.83333333]
    )
    assert result.arrays["dummy_forces"] == approx(
        np.array(
            [
                [0.33333333, 0.83333333, 1.33333333],
                [1.83333333, 2.33333333, 2.83333333],
                [3.33333333, 3.83333333, 4.33333333],
                [4.83333333, 5.33333333, 5.83333333],
                [6.33333333, 6.83333333, 7.33333333],
                [7.83333333, 8.33333333, 8.83333333],
                [9.33333333, 9.83333333, 10.33333333],
                [10.83333333, 11.33333333, 11.83333333],
            ]
        )
    )


def test_spin_mc_boltzmann(sample_atoms):
    obj = CastepSpinMonteCarlo(
        atoms=bulk("SiC", crystalstructure="zincblende", a=4.36, cubic=True),
        method="boltzmann",
        output_prefix="dummy_",
        boltzmann_t=30000,
    )

    result = obj.choose_result(sample_atoms)
    # this has a copy with changed values
    assert result == sample_atoms[0]
    assert result is not sample_atoms[0]
    assert result.info["dummy_energy"] == approx(-9.090663280483277)
    assert result.info["dummy_stress"] == approx(
        [0.0, 0.65844625, 1.31689251, 1.97533876, 2.63378501, 3.29223126]
    )
    assert result.arrays["dummy_forces"] == approx(
        np.array(
            [
                [0.04364373, 0.59077075, 1.13789778],
                [1.68502481, 2.23215183, 2.77927886],
                [3.32640588, 3.87353291, 4.42065994],
                [4.96778696, 5.51491399, 6.06204102],
                [6.60916804, 7.15629507, 7.70342209],
                [8.25054912, 8.79767615, 9.34480317],
                [9.8919302, 10.43905723, 10.98618425],
                [11.53331128, 12.0804383, 12.62756533],
            ]
        )
    )


def test_spin_mc_raise():
    with raises(ValueError):
        obj = CastepSpinMonteCarlo(
            atoms=bulk(
                "SiC", crystalstructure="zincblende", a=4.36, cubic=True
            ),
            method="some unknown one",
        )
