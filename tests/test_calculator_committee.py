import numpy as np
from ase.build import molecule
from ase.calculators.lj import LennardJones
from ase.calculators.morse import MorsePotential
from pytest import approx, raises

from wfl.calculators.committee import calculate_committee

ref_lj_energy = -4.52573996914352
ref_morse_energy = -3.4187397762024867

ref_lj_forces = np.array([[0., 0., 0.],
                          [0.91628394, 0.91628394, 0.91628394],
                          [-0.91628394, -0.91628394, 0.91628394],
                          [0.91628394, -0.91628394, -0.91628394],
                          [-0.91628394, 0.91628394, -0.91628394]])
ref_morse_forces = np.array([[0., 0., 0.],
                             [-1.83980777, -1.83980777, -1.83980777],
                             [1.83980777, 1.83980777, -1.83980777],
                             [-1.83980777, 1.83980777, 1.83980777],
                             [1.83980777, -1.83980777, 1.83980777]])


def test_calculate_committee(tmp_path):
    calculators = [LennardJones(), MorsePotential()]

    with raises(ValueError, match="Prefix with formatting is incorrect.*"):
        _ = calculate_committee(molecule("CH4"), calculators, output_prefix="prefix_with{}_two_formatters{}")

    with raises(ValueError, match="Don't know where to put property.*"):
        _ = calculate_committee(molecule("CH4"), calculators, properties=["energy", "something_we_dont_know"])

    # default prefix
    results0 = calculate_committee(molecule("CH4"), calculators, properties=['energy', 'forces'])
    assert results0.info["committee_0_energy"] == approx(ref_lj_energy)
    assert results0.info["committee_1_energy"] == approx(ref_morse_energy)
    assert results0.arrays["committee_0_forces"] == approx(ref_lj_forces)
    assert results0.arrays["committee_1_forces"] == approx(ref_morse_forces)

    # formatter prefix
    results1 = calculate_committee(molecule("CH4"), calculators, output_prefix="_{}formatter--",
                                   properties=['energy', 'forces'])
    assert results1.info["_0formatter--energy"] == approx(ref_lj_energy)
    assert results1.info["_1formatter--energy"] == approx(ref_morse_energy)
    assert results1.arrays["_0formatter--forces"] == approx(ref_lj_forces)
    assert results1.arrays["_1formatter--forces"] == approx(ref_morse_forces)

    # non-formatter prefix
    results2 = calculate_committee(molecule("CH4"), calculators, output_prefix="general_str",
                                   properties=['energy', 'forces'])
    assert results2.info["general_str0energy"] == approx(ref_lj_energy)
    assert results2.info["general_str1energy"] == approx(ref_morse_energy)
    assert results2.arrays["general_str0forces"] == approx(ref_lj_forces)
    assert results2.arrays["general_str1forces"] == approx(ref_morse_forces)

    # list as input
    results3 = calculate_committee([molecule("CH4"), molecule("CH3")], calculators, properties=['energy', 'forces'])
    assert len(results3) == 2
    assert results3[0].info["committee_0_energy"] == approx(ref_lj_energy)
    assert results3[0].info["committee_1_energy"] == approx(ref_morse_energy)
    assert results3[0].arrays["committee_0_forces"] == approx(ref_lj_forces)
    assert results3[0].arrays["committee_1_forces"] == approx(ref_morse_forces)
