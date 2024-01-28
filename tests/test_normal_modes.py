import numpy as np
import os
import ase.io
import pytest 
from pytest import approx

from wfl.generate import normal_modes as nm 

def test_getting_normal_modes():
    ref_path = os.path.join(os.path.abspath(os.path.dirname(__file__)),
                            'assets/normal_modes/')

    at = ase.io.read(os.path.join(ref_path, 'water_dftb_nms.xyz'))

    my_vib = nm.NormalModes(at, prop_prefix='REF_')


    # test that displacements are done ok
    ref_displaced_ats = ase.io.read(os.path.join(ref_path, 'displaced_out.xyz'), ':')
    vib_displaced_ats = my_vib._displace_at_in_xyz()
    for ref_at, my_at in zip(ref_displaced_ats, vib_displaced_ats):
        assert ref_at.info['disp_direction'] == my_at.info['disp_direction']
        assert np.all(ref_at.positions == approx(my_at.positions))

    # test that numerical normal modes are derived ok
    my_vib._write_nm_to_atoms(displaced_ats=ref_displaced_ats)

    ref_evals = np.array([-9.80822573e-03, -1.13790069e-06, -1.95746658e-09, 2.88386017e-07,
                          2.07461576e-03, 2.12618732e-03, 6.96424592e+00, 4.66006089e+01,
                          5.43176270e+01])

    # test that eigenvalues match ok
    my_evals = my_vib.eigenvalues
    # non-zero modes should match well
    comparison_harm_modes = np.array([ref == approx(my) for ref, my in zip(ref_evals, my_evals)])[
                            6:]
    # zero modes are more messy
    comparison_zero_modes = np.array(
        [ref == approx(my, rel=1e-3, abs=1e-6) for ref, my in zip(ref_evals, my_evals)])[:6]
    assert np.all(comparison_harm_modes)
    assert np.all(comparison_zero_modes)

    # check that eigenvectors match ok
    ref_evecs = np.load(os.path.join(ref_path, 'water_dftb_evecs.npy'))
    my_evecs = my_vib.eigenvectors

    comparison_harm_modes = np.array(
        [approx(abs(np.dot(ref, my))) == 1 for ref, my in zip(ref_evecs, my_evecs)])[6:]
    comparison_zero_modes = np.array(
        [approx(abs(np.dot(ref, my))) == 1 for ref, my in zip(ref_evecs, my_evecs)])[:6]

    assert np.all(comparison_harm_modes)
    assert np.all(comparison_zero_modes)


def test_sample_normal_modes():
    nm_fn = os.path.join(os.path.abspath(os.path.dirname(__file__)),
                         'assets/normal_modes/water_dftb_nms.xyz')

    my_vib = nm.NormalModes(nm_fn, prop_prefix='REF_')
    temp = 300
    at = my_vib.sample_normal_modes(temp=temp, sample_size=1)[0]
    info = list(at.info.keys())

    assert len(info) == 2
    assert 'REF_normal_mode_energy' in info
    assert approx(at.info['REF_normal_mode_temperature']) == temp

    at = my_vib.sample_normal_modes(temp=300, sample_size=1,
                           info_to_keep='free_energy')[0]

    assert len(list(at.info.keys())) == 2

    # test sampling only one normal mode and alternative arrays/info keys
    fake_array = ['a'] * my_vib.num_at
    my_vib.atoms.arrays['fake_array'] = fake_array
    my_vib.atoms.info['fake_info_1'] = 'fake_info_1'
    my_vib.atoms.info['fake_info_2'] = 'fake_info_2'
    info_to_keep = "fake_info_1 fake_info_2"
    at = my_vib.sample_normal_modes(normal_mode_numbers=7, temp=temp, sample_size=1,
                            info_to_keep=info_to_keep,
                            arrays_to_keep="fake_array")[0]

    n_sample = 5
    temp = 300
    sample = my_vib.sample_normal_modes(temp=temp, sample_size=n_sample)
    assert len(sample) == n_sample


def test_view(tmp_path):
    ref_path = os.path.join(os.path.abspath(os.path.dirname(__file__)),
                            'assets/normal_modes/')

    nm_fn = os.path.join(ref_path, 'water_dftb_nms.xyz')
    my_vib = nm.NormalModes(nm_fn, prop_prefix='REF_')

    my_vib.view(output_dir=tmp_path, normal_mode_numbers=7)
    my_mode = ase.io.read(os.path.join(tmp_path, 'nm_7.xyz'), ':')
    ref_mode = ase.io.read(os.path.join(ref_path, 'nm_7.xyz'), ':')

    for my_at, ref_at in zip(my_mode, ref_mode):
        assert np.all(ref_at.positions == approx(my_at.positions))


def test_print_summary():
    nm_fn = os.path.join(os.path.abspath(os.path.dirname(__file__)),
                         'assets/normal_modes/water_dftb_nms.xyz')
    my_vib = nm.NormalModes(nm_fn, prop_prefix='REF_')

    # just check that it prints successfully
    my_vib.summary()
