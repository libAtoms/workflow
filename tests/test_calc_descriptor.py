# NEED TESTS OF LOCAL, RATHER THAN PER-CONFIG AVERAGE, DESCRIPTORS

import numpy as np
import pytest
from ase.atoms import Atoms
from pytest import approx

from wfl.configset import ConfigSet, OutputSpec

try:
    from wfl.descriptors.quippy import calculate
    from quippy.descriptors import Descriptor
except ModuleNotFoundError:
    pytestmark = pytest.mark.skip(reason='no quippy')


def get_ats():
    np.random.seed(5)
    return [Atoms('Si3C1', cell=(2, 2, 2), pbc=[True] * 3, scaled_positions=np.random.uniform(size=(4, 3)))]


def test_calc_descriptor_average_any_atomic_number():
    ats = get_ats()
    ci = ConfigSet(ats)

    ats_desc = calculate(ci, OutputSpec(),
                    'soap n_max=4 l_max=4 cutoff=5.0 atom_sigma=0.5 average n_species=2 species_Z={6 14}', 'desc',
                    per_atom=False)

    d = Descriptor(
        'soap n_max=4 l_max=4 cutoff=5.0 atom_sigma=0.5 average n_species=2 species_Z={6 14}').calc(ats[0])['data'][0]
    assert d.shape[0] == 181

    # check shapes
    for at in ats_desc:
        assert 'desc' in at.info and at.info['desc'].shape == (181,)

    # check one manually
    assert d == approx(list(ats_desc)[0].info['desc'])


def test_calc_descriptor_average_any_atomic_number_normalization():
    ats = get_ats()
    ci = ConfigSet(ats)

    ats_desc = calculate(ci, OutputSpec(), 
                    ['soap n_max=4 l_max=4 cutoff=5.0 atom_sigma=0.5 average n_species=2 species_Z={6 14}', 
                     'soap n_max=3 l_max=3 cutoff=5.0 atom_sigma=0.5 average n_species=2 species_Z={6 14}'], 'desc',
                    per_atom=False)

    assert 1.0 == approx(np.linalg.norm(list(ats_desc)[0].info['desc']))

    ats[0].info.pop("desc", None)
    ats_desc = calculate(ci, OutputSpec(),
                    ['soap n_max=4 l_max=4 cutoff=5.0 atom_sigma=0.5 average n_species=2 species_Z={6 14}', 
                     'soap n_max=3 l_max=3 cutoff=5.0 atom_sigma=0.5 average n_species=2 species_Z={6 14}'], 'desc', normalize=False,
                    per_atom=False)
    assert np.sqrt(2) == approx(np.linalg.norm(list(ats_desc)[0].info['desc']))


def test_calc_descriptor_average_z_specific():
    ats = get_ats()
    ci = ConfigSet(ats)

    descs = {6: 'soap n_max=4 l_max=4 cutoff=5.0 atom_sigma=0.5 average Z=6 n_species=2 species_Z={6 14}',
             14: 'soap n_max=4 l_max=3 cutoff=5.0 atom_sigma=0.5 average Z=14 n_species=2 species_Z={6 14}'}

    ats_desc = calculate(ci, OutputSpec(), descs, 'desc', per_atom=False)

    d6 = Descriptor(descs[6]).calc(ats[0])['data'][0]
    d14 = Descriptor(descs[14]).calc(ats[0])['data'][0]
    assert d6.shape[0] + d14.shape[0] == 326

    # check shape
    for at in ats_desc:
        assert 'desc' in at.info
        assert at.info['desc'].shape == (326,)

    # check against manually composition-weighted and normalized vector
    desc_manual = list(d6 * 0.25) + list(d14 * 0.75)
    desc_manual /= np.linalg.norm(desc_manual)

    assert desc_manual == approx(list(ats_desc)[0].info['desc'])


def test_calc_descriptor_any_atomic_number():
    ats = get_ats()
    ci = ConfigSet(ats)

    ats_desc = calculate(ci, OutputSpec(),
                    'soap n_max=4 l_max=4 cutoff=5.0 atom_sigma=0.5 n_species=2 species_Z={6 14}', 'desc',
                    per_atom=True)

    d = Descriptor(
        'soap n_max=4 l_max=4 cutoff=5.0 atom_sigma=0.5 n_species=2 species_Z={6 14}').calc(ats[0])['data']
    print("d.shape", d.shape)
    assert d.shape == (4, 181)

    # check shapes
    for at in ats_desc:
        assert 'desc' in at.arrays and at.arrays['desc'].shape == (len(at), 181)

    # check one manually
    assert d == approx(list(ats_desc)[0].arrays['desc'])


def test_calc_descriptor_any_atomic_number_normalization():
    ats = get_ats()
    ci = ConfigSet(ats)

    ats_desc = calculate(ci, OutputSpec(), 
                    ['soap n_max=4 l_max=4 cutoff=5.0 atom_sigma=0.5 n_species=2 species_Z={6 14}', 
                     'soap n_max=3 l_max=3 cutoff=5.0 atom_sigma=0.5 n_species=2 species_Z={6 14}'], 'desc',
                    per_atom=True)

    at = list(ats_desc)[0]
    assert 1.0 == approx(np.linalg.norm(at.arrays['desc'], axis=1))

    del at.arrays["desc"]

    ats_desc = calculate(ci, OutputSpec(), 
                    ['soap n_max=4 l_max=4 cutoff=5.0 atom_sigma=0.5 n_species=2 species_Z={6 14}', 
                     'soap n_max=3 l_max=3 cutoff=5.0 atom_sigma=0.5 n_species=2 species_Z={6 14}'], 'desc',
                    per_atom=True, normalize=False)

    at = list(ats_desc)[0]
    assert np.sqrt(2.0) == approx(np.linalg.norm(at.arrays['desc'], axis=1))


def test_calc_descriptor_z_specific():
    ats = get_ats()
    ci = ConfigSet(ats)

    descs = {6: 'soap n_max=4 l_max=4 cutoff=5.0 atom_sigma=0.5 Z=6 n_species=2 species_Z={6 14}',
             14: 'soap n_max=4 l_max=3 cutoff=5.0 atom_sigma=0.5 Z=14 n_species=2 species_Z={6 14}'}

    ats_desc = calculate(ci, OutputSpec(), descs, 'desc', per_atom=True)

    d6 = Descriptor(descs[6]).calc(ats[0])['data']
    d14 = Descriptor(descs[14]).calc(ats[0])['data']
    assert d6.shape == (1, 181)
    assert d14.shape == (3, 145)

    # check shape
    for at in ats_desc:
        assert 'desc_Z_6' in at.arrays
        assert at.arrays['desc_Z_6'].shape == (len(at),181)
        assert 'desc_Z_14' in at.arrays
        assert at.arrays['desc_Z_14'].shape == (len(at),145)

    # check against manually computed vectors

    at = list(ats_desc)[0]
    assert d6 == approx(at.arrays['desc_Z_6'][at.numbers == 6])
    assert d14 == approx(at.arrays['desc_Z_14'][at.numbers == 14])
