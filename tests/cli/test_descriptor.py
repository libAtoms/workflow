import os
import numpy as np

import click
from click.testing import CliRunner

import pytest
from pytest import approx

from ase import Atoms
from ase.io import write, read

from wfl.cli.cli import cli

try:
    from quippy.descriptors import Descriptor
except ModuleNotFoundError:
    pytestmark = pytest.mark.skip(reason='no quippy')


def get_ats():
    np.random.seed(5)
    return [Atoms('Si3C1', cell=(2, 2, 2), pbc=[True] * 3, scaled_positions=np.random.uniform(size=(4, 3)))]


def test_descriptor_quippy(tmp_path):
    ats = get_ats()
    
    fn_in = tmp_path / "ats_in.xyz" 
    fn_out = tmp_path / "ats_out.xyz"

    write(fn_in, ats)

    descriptor_str = 'soap n_max=4 l_max=4 cutoff=5.0 atom_sigma=0.5 average n_species=2 species_Z={6 14}'

    params = [
        '-v',
        'descriptor', 
        'quippy', 
        f'-i {str(fn_in)}',
        f'-o {str(fn_out)}',
        f'--descriptor "{descriptor_str}" ',
        f'--key soap'
    ]

    runner = CliRunner()
    result = runner.invoke(cli, ' '.join(params))

    target = Descriptor(
        descriptor_str).calc(ats[0])['data'][0]
    assert target.shape[0] == 181

    ats_out = read(fn_out, ":")

    #check that shape matches
    for at in ats_out:
        assert 'soap' in at.info and at.info['soap'].shape == (181,) 

    # check one manually
    assert target == approx(list(ats_out)[0].info['soap'])
