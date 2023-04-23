import os
import warnings

import numpy as np
import pytest
from ase.atoms import Atoms
from ase.calculators.lj import LennardJones
from pytest import approx

from pprint import pprint

from wfl.calculators.generic import calculate as generic_calc
from wfl.configset import ConfigSet, OutputSpec
from wfl.fit.error import calc as ref_err_calc
from wfl.fit.error import value_error_scatter as plot_error
from wfl.utils.misc import dict_tuple_keys_to_str

# values to check against
__ALL__energy = 1085377319.8139253
__ALL__forces = 51078415139.79738
__ALL__energy_per_atom = __ALL__energy / 4.0
__ALL__virial = 4661857254.028029
__ALL__virial_per_atom = __ALL__virial / 4.0
__ALL__stress = 72841519.59418797

__0__virial_per_atom = 6903213.131588299
__0__forces = 177310599.2955574
__0__energy_per_atom = 1634085.82578905

__1__virial_per_atom = 2099337656.2681184
__1__forces = 92542999106.76888
__1__energy_per_atom = 490206396.9033798

__None__virial_per_atom = 300529496.24497306
__None__forces = 9966576189.478043
__None__energy_per_atom = 61972793.576289155


@pytest.fixture
def ref_atoms():
    np.random.seed(6)

    ats = [Atoms(cell=[[4, 0, 0], [0, 4, 0], [0, 0, 4]], positions=np.random.uniform(size=(4, 3)),
                 numbers=[13] * 4, pbc=[True] * 3) for _ in range(10)]
    for at_i, at in enumerate(ats):
        if at_i < 6:
            at.info['category'] = at_i // 3
        at.info['subcategory'] = at_i % 2

    ci = ConfigSet(ats)

    ats_eval = generic_calc(ci, OutputSpec(), LennardJones(sigma=1.0), output_prefix='REF_')

    return ats_eval


def test_error(tmp_path, ref_atoms):
    calc_ats = generic_calc(ConfigSet(ref_atoms), OutputSpec(), LennardJones(sigma=0.75), output_prefix='LJ_')
    ref_err_dict, _, _ = ref_err_calc(calc_ats, 'LJ_', 'REF_', category_keys='category')

    print("test_error")
    pprint(ref_err_dict)
    assert ref_err_dict["forces"].keys() == {'_ALL_', '0', '1', 'None'}

    assert approx(ref_err_dict['energy/atom']["_ALL_"]['RMSE']) == __ALL__energy_per_atom
    assert approx(ref_err_dict['forces']['_ALL_']['RMSE']) == __ALL__forces
    assert approx(ref_err_dict['virial/atom/comp']['_ALL_']['RMSE']) == __ALL__virial_per_atom
    assert ref_err_dict['energy/atom']['_ALL_']["count"] == 10
    assert ref_err_dict['forces']['_ALL_']["count"] == 10 * 4
    assert ref_err_dict['virial/atom/comp']['_ALL_']["count"] == 10 * 6

    assert approx(ref_err_dict['energy/atom']["0"]['RMSE']) == __0__energy_per_atom
    assert approx(ref_err_dict['forces']["0"]['RMSE']) == __0__forces
    assert approx(ref_err_dict['virial/atom/comp']["0"]['RMSE']) == __0__virial_per_atom
    assert ref_err_dict['energy/atom']["0"]["count"] == 3
    assert ref_err_dict['forces']["0"]["count"] == 3 * 4
    assert ref_err_dict['virial/atom/comp']["0"]["count"] == 3 * 6

    assert approx(ref_err_dict['energy/atom']["1"]['RMSE']) == __1__energy_per_atom
    assert approx(ref_err_dict['forces']["1"]['RMSE']) == __1__forces
    assert approx(ref_err_dict['virial/atom/comp']["1"]['RMSE']) == __1__virial_per_atom
    assert ref_err_dict['energy/atom']["1"]["count"] == 3
    assert ref_err_dict['forces']["1"]["count"] == 3 * 4
    assert ref_err_dict['virial/atom/comp']["1"]["count"] == 3 * 6

    assert approx(ref_err_dict['energy/atom']["None"]['RMSE']) == __None__energy_per_atom
    assert approx(ref_err_dict['forces']["None"]['RMSE']) == __None__forces
    assert approx(ref_err_dict['virial/atom/comp']["None"]['RMSE']) == __None__virial_per_atom
    assert ref_err_dict['energy/atom']["None"]["count"] == 4
    assert ref_err_dict['forces']["None"]["count"] == 4 * 4
    assert ref_err_dict['virial/atom/comp']["None"]["count"] == 4 * 6


def test_err_from_calc(ref_atoms):
    ref_atoms_calc = generic_calc(ref_atoms, OutputSpec(), LennardJones(sigma=0.75), output_prefix='calc_')
    ref_err_dict, _, _ = ref_err_calc(ref_atoms_calc, ref_property_prefix='REF_', calc_property_prefix='calc_')

    assert approx(ref_err_dict['energy/atom']['_ALL_']["RMSE"]) == __ALL__energy_per_atom
    assert approx(ref_err_dict['forces']['_ALL_']["RMSE"]) == __ALL__forces
    assert approx(ref_err_dict['virial/atom/comp']['_ALL_']["RMSE"]) == __ALL__virial_per_atom
    assert ref_err_dict['energy/atom']['_ALL_']["count"] == 10
    assert ref_err_dict['forces']['_ALL_']["count"] == 10 * 4
    assert ref_err_dict['virial/atom/comp']['_ALL_']["count"] == 10 * 6


def test_error_properties(ref_atoms):
    ref_atoms_calc = generic_calc(ref_atoms, OutputSpec(), LennardJones(sigma=0.75), output_prefix='calc_')
    # both energy and per atom
    ref_err_dict, _, _ = ref_err_calc(
        ref_atoms_calc, ref_property_prefix='REF_', calc_property_prefix='calc_', config_properties=["energy", "energy/atom"],
        category_keys=None)
    assert len(ref_err_dict.keys()) == 2
    assert len(ref_err_dict['energy'].keys()) == 2

    assert approx(ref_err_dict['energy/atom']['_ALL_']["RMSE"]) == __ALL__energy_per_atom
    assert approx(ref_err_dict['energy']['_ALL_']["RMSE"]) == __ALL__energy

    assert ref_err_dict['energy']['_ALL_']["count"] == 10
    assert ref_err_dict['energy/atom']['_ALL_']["count"] == 10

    # only energy
    ref_err_dict, _, _ = ref_err_calc(
        ref_atoms_calc, ref_property_prefix='REF_', calc_property_prefix='calc_', config_properties=["energy"],
        category_keys=None)

    assert len(ref_err_dict.keys()) == 1
    assert len(ref_err_dict["energy"].keys()) == 2

    assert ref_err_dict['energy']['_ALL_']["count"] == 10
    assert approx(ref_err_dict['energy']['_ALL_']["RMSE"]) == __ALL__energy

    # only stress
    ref_err_dict, _, _ = ref_err_calc(
        ref_atoms_calc, ref_property_prefix='REF_', calc_property_prefix='calc_', config_properties=["stress/comp", "virial/comp"],
        category_keys=None)

    assert len(ref_err_dict.keys()) == 2
    assert len(ref_err_dict["stress/comp"].keys()) == 2

    assert ref_err_dict['stress/comp']['_ALL_']["count"] == 60
    assert approx(ref_err_dict['stress/comp']['_ALL_']["RMSE"]) == __ALL__stress

    assert ref_err_dict['virial/comp']['_ALL_']["count"] == 60
    assert approx(ref_err_dict['virial/comp']['_ALL_']["RMSE"]) == __ALL__virial


def test_error_forces(ref_atoms):
    ref_atoms_calc = generic_calc(ref_atoms, OutputSpec(), LennardJones(sigma=0.75), output_prefix='calc_')

    # forces by element
    ref_err_dict, _, _ = ref_err_calc(
        ref_atoms_calc, ref_property_prefix='REF_', calc_property_prefix='calc_', atom_properties=["forces/Z"])
    assert ref_err_dict['forces/Z_13']['_ALL_']["count"] == 40
    assert approx(ref_err_dict['forces/Z_13']['_ALL_']["RMSE"]) == __ALL__forces

    # remove error info so next call won't complain
    for at in ref_atoms_calc:
        at.info = {k: v for k, v in at.info.items() if not k.endswith('_error')}
    for at in ref_atoms_calc:
        at.arrays = {k: v for k, v in at.arrays.items() if not k.endswith('_error')}

    # forces by component
    ref_err_dict, _, _ = ref_err_calc(
        ref_atoms_calc, ref_property_prefix='REF_', calc_property_prefix='calc_', atom_properties=["forces/comp/Z"])
    assert ref_err_dict['forces/comp/Z_13']['_ALL_']["count"] == 120
    assert approx(ref_err_dict['forces/comp/Z_13']['_ALL_']["RMSE"]) == 29490136730.741474


def test_error_missing(ref_atoms):
    ref_atoms_calc = generic_calc(ref_atoms, OutputSpec(), LennardJones(sigma=0.75), output_prefix='calc_')

    ref_atoms_list = [at for at in ref_atoms_calc]

    for at in ref_atoms_list:
        if at.info.get("category", None) == 0:
            del at.info["REF_energy"]
        if at.info.get("category", None) == 1:
            del at.info["REF_stress"]
        if at.info.get("category", None) is None:
            del at.arrays["REF_forces"]

    # forces by element
    ref_err_dict, _, _ = ref_err_calc(
        ref_atoms_list, ref_property_prefix='REF_', calc_property_prefix='calc_', category_keys=[],
        config_properties=["energy/atom", "virial/atom"], atom_properties=["forces"])

    assert ref_err_dict["energy/atom"]["_ALL_"]["count"] == 7
    assert ref_err_dict["forces"]["_ALL_"]["count"] == 24
    assert ref_err_dict["virial/atom"]["_ALL_"]["count"] == 7


def test_error_subcat(ref_atoms):
    ref_atoms_calc = generic_calc(ref_atoms, OutputSpec(), LennardJones(sigma=0.75), output_prefix='calc_')

    ref_atoms_list = [at for at in ref_atoms_calc]

    # forces by element
    ref_err_dict, _, _ = ref_err_calc(
        ref_atoms_list, ref_property_prefix='REF_', calc_property_prefix='calc_', category_keys=["category", "subcategory"],
        config_properties=["energy/atom", "virial/atom"], atom_properties=["forces"])

    assert ref_err_dict["energy/atom"]["_ALL_"]["count"] == 10
    assert ref_err_dict["forces"]["_ALL_"]["count"] == 40
    assert ref_err_dict["virial/atom"]["_ALL_"]["count"] == 10

    assert ref_err_dict["energy/atom"]["0 / 0"]["count"] == 2
    assert ref_err_dict["energy/atom"]["0 / 1"]["count"] == 1
    assert ref_err_dict["energy/atom"]["1 / 0"]["count"] == 1
    assert ref_err_dict["energy/atom"]["1 / 1"]["count"] == 2
    assert ref_err_dict["energy/atom"]["None / 0"]["count"] == 2
    assert ref_err_dict["energy/atom"]["None / 1"]["count"] == 2


def test_error_diffs(tmp_path, ref_atoms):
    calc_ats = generic_calc(ConfigSet(ref_atoms), OutputSpec(), LennardJones(sigma=0.75), output_prefix='LJ_')
    error_dict, error_diffs, error_parity = ref_err_calc(calc_ats, 'LJ_', 'REF_', category_keys='category')

    assert error_dict["virial/atom/comp"]["None"]["count"] == 24
    assert len(error_diffs["virial/atom/comp"]["None"]) == 24
    assert len(error_parity["ref"]["virial/atom/comp"]["None"]) == 24
    assert len(error_parity["calc"]["virial/atom/comp"]["None"]) == 24


def test_plot_error(tmp_path):
    filename = os.path.join(os.path.abspath(os.path.dirname(__file__)),  'assets', 'configs_for_error_test.xyz')
    inputs = ConfigSet(filename)

    errors, diffs, parity = ref_err_calc(
        inputs=inputs, 
        calc_property_prefix='mace_',
        ref_property_prefix='dft_',
        config_properties=["energy/atom", "energy"],
        atom_properties=["forces/comp/Z", "forces/comp"],
        category_keys="mol_or_rad"
    )

    plot_error(errors, diffs, parity, ref_property_prefix='dft_', calc_property_prefix='mace_', output=tmp_path/"error_plot_both.png")
    plot_error(errors, diffs, parity, properties=["energy", "energy/atom"],
               ref_property_prefix='dft_', calc_property_prefix='mace_', 
               cmap="tab10", output=tmp_path/"error_plot_selected_both.png")

    plot_error(errors, diffs, parity, ref_property_prefix='dft_', calc_property_prefix='mace_', output=tmp_path/"error_plot_parity_only.png", plot_parity=True, plot_error=False)
    plot_error(errors, diffs, parity, ref_property_prefix='dft_', calc_property_prefix='mace_', output=tmp_path/"error_plot_error_only.png", plot_parity=False, plot_error=True)

    warnings.warn(f"error plots in {tmp_path}/error_plot*.png")

