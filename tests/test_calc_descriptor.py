# NEED TESTS OF LOCAL, RATHER THAN PER-CONFIG AVERAGE, DESCRIPTORS

import numpy as np
import pytest
from ase.atoms import Atoms
from pytest import approx

from wfl.configset import ConfigSet, OutputSpec

try:
    from wfl.descriptors.calc import calc
    from quippy.descriptors import Descriptor
except ModuleNotFoundError:
    pytestmark = pytest.mark.skip(reason='no quippy')


def get_ats():
    np.random.seed(5)
    return [Atoms('Si3C1', cell=(2, 2, 2), pbc=[True] * 3, scaled_positions=np.random.uniform(size=(4, 3)))]


def test_calc_descriptor_average_any_atomic_number():
    ats = get_ats()
    ci = ConfigSet(input_configs=ats)

    ats_desc = calc(ci, OutputSpec(),
                    'soap n_max=4 l_max=4 cutoff=5.0 atom_sigma=0.5 average n_species=2 species_Z={6 14}', 'desc')

    d = Descriptor(
        'soap n_max=4 l_max=4 cutoff=5.0 atom_sigma=0.5 average n_species=2 species_Z={6 14}').calc(ats[0])['data'][0]
    assert d.shape[0] == 181

    # check shapes
    for at in ats_desc:
        assert 'desc' in at.info and at.info['desc'].shape == (181,)

    # check one manually
    assert d == approx(list(ats_desc)[0].info['desc'])


def test_calc_descriptor_average_z_specific():
    ats = get_ats()
    ci = ConfigSet(input_configs=ats)

    descs = {6: 'soap n_max=4 l_max=4 cutoff=5.0 atom_sigma=0.5 average Z=6 n_species=2 species_Z={6 14}',
             14: 'soap n_max=4 l_max=3 cutoff=5.0 atom_sigma=0.5 average Z=14 n_species=2 species_Z={6 14}'}

    ats_desc = calc(ci, OutputSpec(), descs, 'desc')

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
