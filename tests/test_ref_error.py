import numpy as np
import pytest
from ase.atoms import Atoms
from ase.calculators.lj import LennardJones
from pytest import approx

from pprint import pprint

from wfl.calculators.generic import run as generic_calc
from wfl.configset import ConfigSet, OutputSpec
from wfl.fit.ref_error import calc as ref_err_calc
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


def test_ref_error(tmp_path, ref_atoms):
    ref_err_dict = ref_err_calc(ref_atoms, LennardJones(sigma=0.75), 'REF_',
                                category_keys='category', config_properties=["energy/atom", "virial/atom/comp"],
                                atom_properties=["forces"])

    pprint(ref_err_dict)
    assert ref_err_dict.keys() == {'_ALL_', '0', '1', 'None'}

    assert approx(ref_err_dict['_ALL_']['energy/atom']['RMS']) == __ALL__energy_per_atom
    assert approx(ref_err_dict['_ALL_']['forces']['RMS']) == __ALL__forces
    assert approx(ref_err_dict['_ALL_']['virial/atom/comp']['RMS']) == __ALL__virial_per_atom
    assert ref_err_dict['_ALL_']['energy/atom']['num'] == 10
    assert ref_err_dict['_ALL_']['forces']['num'] == 10 * 4
    assert ref_err_dict['_ALL_']['virial/atom/comp']['num'] == 10 * 6

    assert approx(ref_err_dict["0"]['energy/atom']['RMS']) == __0__energy_per_atom
    assert approx(ref_err_dict["0"]['forces']['RMS']) == __0__forces
    assert approx(ref_err_dict["0"]['virial/atom/comp']['RMS']) == __0__virial_per_atom
    assert ref_err_dict["0"]['energy/atom']['num'] == 3
    assert ref_err_dict["0"]['forces']['num'] == 3 * 4
    assert ref_err_dict["0"]['virial/atom/comp']['num'] == 3 * 6

    assert approx(ref_err_dict["1"]['energy/atom']['RMS']) == __1__energy_per_atom
    assert approx(ref_err_dict["1"]['forces']['RMS']) == __1__forces
    assert approx(ref_err_dict["1"]['virial/atom/comp']['RMS']) == __1__virial_per_atom
    assert ref_err_dict["1"]['energy/atom']['num'] == 3
    assert ref_err_dict["1"]['forces']['num'] == 3 * 4
    assert ref_err_dict["1"]['virial/atom/comp']['num'] == 3 * 6

    assert approx(ref_err_dict["None"]['energy/atom']['RMS']) == __None__energy_per_atom
    assert approx(ref_err_dict["None"]['forces']['RMS']) == __None__forces
    assert approx(ref_err_dict["None"]['virial/atom/comp']['RMS']) == __None__virial_per_atom
    assert ref_err_dict["None"]['energy/atom']['num'] == 4
    assert ref_err_dict["None"]['forces']['num'] == 4 * 4
    assert ref_err_dict["None"]['virial/atom/comp']['num'] == 4 * 6


def test_err_from_calc(ref_atoms):
    ref_atoms_calc = calc_run(ref_atoms, LennardJones(sigma=0.75), output_prefix='calc_')
    ref_err_dict = ref_erro_calc(ref_atoms_calc, ref_property_prefix='REF_', calc_property_prefix='calc_',
                                 properties=['energy/atom', 'forces', 'virial/atom'])

    print(ref_err_dict)
    assert approx(ref_err_dict['_ALL_']['energy/atom'][1]) == __ALL__energy_per_atom
    assert approx(ref_err_dict['_ALL_']['forces'][1]) == __ALL__forces
    assert approx(ref_err_dict['_ALL_']['virial/atom'][1]) == __ALL__virial_per_atom
    assert ref_err_dict['_ALL_']['energy/atom'][0] == 10
    assert ref_err_dict['_ALL_']['forces'][0] == 10 * 4
    assert ref_err_dict['_ALL_']['virial/atom'][0] == 10 * 6


def test_ref_error_properties(ref_atoms):
    ref_atoms_calc = calc_run(ref_atoms, LennardJones(sigma=0.75), output_prefix='calc_')

    # both energy and per atom
    ref_err_dict = ref_err_calc(
        ref_atoms_calc, ref_property_prefix='REF_', calc_property_prefix='calc_', properties=["energy", "energy_per_atom"])

    assert len(ref_err_dict.keys()) == 1
    assert len(ref_err_dict['_ALL_'].keys()) == 2

    assert approx(ref_err_dict['_ALL_']['energy/atom'][1]) == __ALL__energy_per_atom
    assert approx(ref_err_dict['_ALL_']['energy'][1]) == __ALL__energy

    assert ref_err_dict['_ALL_']['energy'][0] == 10
    assert ref_err_dict['_ALL_']['energy/atom'][0] == 10

    # only energy
    ref_err_dict = ref_err_calc(
        ref_atoms_calc, ref_property_prefix='REF_', calc_property_prefix='calc_', properties=["energy"])

    assert len(ref_err_dict.keys()) == 1
    assert len(ref_err_dict['_ALL_'].keys()) == 1

    assert ref_err_dict['_ALL_']['energy'][0] == 10
    assert approx(ref_err_dict['_ALL_']['energy'][1]) == __ALL__energy

    # only stress
    ref_err_dict = ref_err_calc(
        ref_atoms_calc, ref_property_prefix='REF_', calc_property_prefix='calc_', properties=["stress", "virial"])

    assert len(ref_err_dict.keys()) == 1
    assert len(ref_err_dict['_ALL_'].keys()) == 2

    assert ref_err_dict['_ALL_']['stress'][0] == 60
    assert approx(ref_err_dict['_ALL_']['stress'][1]) == __ALL__stress

    assert ref_err_dict['_ALL_']['virial'][0] == 60
    assert approx(ref_err_dict['_ALL_']['virial'][1]) == __ALL__virial


def test_ref_error_forces(ref_atoms):
    ref_atoms_calc = calc_run(ref_atoms, LennardJones(sigma=0.75), output_prefix='calc_')

    # forces by element
    ref_err_dict = ref_err_calc(
        ref_atoms_calc, ref_property_prefix='REF_', calc_property_prefix='calc_', properties=["forces/Z"])
    assert ref_err_dict['_ALL_']['forces'][13][0] == 40
    assert approx(ref_err_dict['_ALL_']['forces'][13][1]) == __ALL__forces

    # remove error info so next call won't complain
    for at in ref_atoms_calc:
        at.info = {k: v for k, v in at.info.items() if not k.endswith('_error')}
    for at in ref_atoms_calc:
        at.arrays = {k: v for k, v in at.arrays.items() if not k.endswith('_error')}

    # forces by component
    ref_err_dict = ref_err_calc(
        ref_atoms_calc, ref_property_prefix='REF_', calc_property_prefix='calc_', properties=["forces/comp/Z"])
    assert ref_err_dict['_ALL_']['forces'][13][0] == 120
    assert approx(ref_err_dict['_ALL_']['forces'][13][1]) == 29490136730.741474


@pytest.mark.filterwarnings("ignore:missing reference:UserWarning")
def test_ref_error_missing(ref_atoms):
    ref_atoms_calc = calc_run(ref_atoms, LennardJones(sigma=0.75), output_prefix='calc_')

    ref_atoms_list = [at for at in ref_atoms_calc]

    for at in ref_atoms_list:
        if at.info.get("category", None) == 0:
            del at.info["REF_energy"]
        if at.info.get("category", None) == 1:
            del at.info["REF_stress"]
        if at.info.get("category", None) is None:
            del at.arrays["REF_forces"]

    # forces by element
    ref_err_dict = ref_err_calc(
        ref_atoms_list, ref_property_prefix='REF_', calc_property_prefix='calc_', category_keys=[])

    assert ref_err_dict["_ALL_"]["energy_per_atom"][0] == 7
    assert ref_err_dict["_ALL_"]["forces"][0] == 24
    assert ref_err_dict["_ALL_"]["virial_per_atom"][0] == 42


def test_dict_tuple_keys_to_str():
    in_dict = {"_just-a-str-key_": 0.12312,
               ("a single element tuple",): 172398,
               ("multi", "element", "tuple"): 1325235,
               ("my value is a dict",): {"a": 34}}

    out_ref = {"_just-a-str-key_": 0.12312,
               "(a single element tuple)": 172398,
               "(multi,element,tuple)": 1325235,
               "(my value is a dict)": {"a": 34}}

    out = dict_tuple_keys_to_str(in_dict)

    print('out_ref')
    pprint(out_ref)
    print('out')
    pprint(out)
    for key, val in out.items():
        print('key', key, 'val', val)
        assert key in out_ref.keys()
        assert out_ref[key] == val
