import numpy as np
from pytest import approx, raises

from wfl.configset import ConfigSet, OutputSpec
from wfl.generate.atoms_and_dimers import prepare, isolated_atom_from_e0


def test_isolated_atom_from_e0(tmp_path):
    e0_dict = dict(H=1.0, C=5.0, As=123.0)
    e0_dict_num = {1: 1.0, 3: 3.0}
    extra_info_faulty = dict(energy=0.0)
    extra_info_ok = dict(some_info="dummy str")

    configset_0 = OutputSpec("test.isolated0.xyz", file_root=tmp_path)
    configset_1 = OutputSpec("test.isolated1.xyz", file_root=tmp_path)

    # expected errors
    with raises(ValueError, match=r".*key given in the extra info for isolated atom.*"):
        isolated_atom_from_e0(None, e0_dict, 10.0, extra_info=extra_info_faulty)

    # normal behaviour
    isolated_atom_from_e0(configset_0, e0_dict, 10.0, "energy_key", extra_info_ok)
    test_in0 = ConfigSet("test.isolated0.xyz", file_root=tmp_path)
    for at in test_in0:
        assert at.info["energy_key"] == e0_dict[at.get_chemical_formula()]
        for key, val in extra_info_ok.items():
            assert at.info[key] == val
        assert np.all(at.cell == [[10.0, 0.0, 0.0], [0.0, 10.0, 0.0], [0.0, 0.0, 10.0]])

    isolated_atom_from_e0(configset_1, e0_dict_num, 10.0)
    test_in1 = ConfigSet("test.isolated1.xyz", file_root=tmp_path)
    for at in test_in1:
        assert at.get_potential_energy() == e0_dict_num[at.get_atomic_numbers()[0]]


def test_dimers(tmp_path):
    atomic_numbers = [1, 6, 50]
    bond_lengths = {1: 1.0, 6: 3.0, 50: 10}

    configset_0 = OutputSpec("test.dimer0.xyz", file_root=tmp_path)
    configset_1 = OutputSpec("test.dimer1.xyz", file_root=tmp_path)

    configset_1p = OutputSpec("test.dimer1p.xyz", file_root=tmp_path)
    configset_2 = OutputSpec("test.dimer2.xyz", file_root=tmp_path)

    # distances as bond lengths' multiple, no single atoms
    prepare(
        configset_0,
        atomic_numbers,
        bond_lengths,
        dimer_n_steps=3,
        dimer_factor_range=(0.5, 2.0),
        dimer_box_scale=10,
        do_isolated_atoms=False,
    )

    # no single atoms created
    assert all(
        [
            at.info.get("config_type", "") != "isolated_atom"
            for at in configset_0.to_ConfigSet()
        ]
    )

    distances_created = [at.get_distance(0, 1) for at in configset_0.to_ConfigSet()]
    volumes_created = [at.get_volume() for at in configset_0.to_ConfigSet()]
    assert np.min(distances_created) == 0.5
    assert np.max(distances_created) == 20.0
    assert len(distances_created) == 18
    assert np.min(volumes_created) == approx(1.0 * 10.0 ** 3)
    assert np.max(volumes_created) == approx((10.0 * 10.0) ** 3)

    # distances as actual distance given
    prepare(
        configset_1,
        atomic_numbers,
        bond_lengths=None,
        dimer_n_steps=3,
        dimer_factor_range=(0.4, 1.6),
        dimer_box_scale=10,
        do_isolated_atoms=True,
        max_cutoff=6.0,
    )
    distances_created = [
        at.get_distance(0, 1) for at in configset_1.to_ConfigSet() if len(at) > 1
    ]
    volume_isolated_atom = [
        at.get_volume()
        for at in configset_1.to_ConfigSet()
        if at.info.get("config_type", "") == "isolated_atom"
    ]
    assert sorted(np.unique(distances_created)) == approx([0.4, 1.0, 1.6])
    # assert sorted(volume_isolated_atom) == approx([10. ** 3, 30. ** 3, 100. ** 3])
    assert volume_isolated_atom == approx([10.0 ** 3] * 3)

    # check max_cutoff expanding beyond bond-related size
    prepare(
        configset_1p,
        atomic_numbers,
        bond_lengths=None,
        dimer_n_steps=3,
        dimer_factor_range=(0.4, 1.6),
        dimer_box_scale=10,
        do_isolated_atoms=True,
        max_cutoff=16.0,
    )
    distances_created = [
        at.get_distance(0, 1) for at in configset_1p.to_ConfigSet() if len(at) > 1
    ]
    volume_isolated_atom = [
        at.get_volume()
        for at in configset_1p.to_ConfigSet()
        if at.info.get("config_type", "") == "isolated_atom"
    ]
    assert sorted(np.unique(distances_created)) == approx([0.4, 1.0, 1.6])
    assert volume_isolated_atom == approx([16.1 ** 3] * 3)

    # fixed cell
    prepare(
        configset_2,
        atomic_numbers,
        bond_lengths=None,
        dimer_n_steps=3,
        dimer_factor_range=(0.4, 1.6),
        do_isolated_atoms=True,
        fixed_cell=[10, 20, 30],
    )
    for at in configset_2.to_ConfigSet():
        assert at.cell.array == approx(np.array([[10, 0, 0], [0, 20, 0], [0, 0, 30]]))
