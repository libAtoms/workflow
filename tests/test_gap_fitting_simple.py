import os
import shutil
import json

from pathlib import Path

import pytest
from click.testing import CliRunner

from wfl.fit.gap import simple as gap_simple
from wfl.utils.quip_cli_strings import dict_to_quip_str

from wfl.cli.cli import cli

def test_dict_to_quip_str():
    # this comparison should probably be done in a way that's less sensitive to whitespace
    descriptor_dict = {'soap': True, 'l_max': 4, 'n_max': 12, 'cutoff': 6,
                       'delta': 1, 'covariance_type': 'dot_product',
                       'zeta': 4, 'n_sparse': 100,
                       'sparse_method': 'cur_points',
                       'config_type_sigma': { 'cfg1' : [1.0, 2.0, 0.3, 4.0], 'cfg2': [1, 1, 1, 1]},
                       'atom_gaussian_width': 0.3, 'add_species': False,
                       'n_species': 3, 'Z': 8, 'species_Z': [8, 1, 6]}

    expected_string_double = 'soap=T l_max=4 n_max=12 cutoff=6 delta=1' \
                             ' covariance_type=dot_product zeta=4 n_sparse=100 ' \
                             'sparse_method=cur_points config_type_sigma=cfg1:1.0:2.0:0.3:4.0:cfg2:1:1:1:1 ' \
                             'atom_gaussian_width=0.3 add_species=F ' \
                             'n_species=3 Z=8 species_Z={{8 1 6}}'

    expected_string_single = expected_string_double.replace('{{', '{').replace('}}', '}')

    descriptor_string_double = dict_to_quip_str(descriptor_dict, list_brackets='{{}}')
    assert descriptor_string_double == expected_string_double

    string_single = dict_to_quip_str(descriptor_dict, '{}')
    assert string_single == expected_string_single

    string_default = dict_to_quip_str(descriptor_dict)
    assert string_default == expected_string_single


def test_dict_to_gap_fit_string():
    # this comparison should probably be done in a way that's less sensitive to whitespace
    gap_fit_dict = {'atoms_filename': 'train.xyz',
                    'default_sigma': [0.01, 0.1, 0.1, 0.0],
                    'sparse_seprate_file': False,
                    'core_ip_args': 'IP Glue',
                    'core_param_file': '/test/path/test/file.xml',
                    'config_type_sigma': 'isolated_atom:1e-05:0.0:0.0:0.0:funky_configs:0.1:0.3:0.0:0.0',
                    '_gap': [
                        {'soap': True, 'l_max': 6, 'n_max': '12',
                         'cutoff': 3, 'delta': 1,
                         'covariance_type': 'dot_product', 'zeta': 4,
                         'n_sparse': 200, 'sparse_method': 'cur_points'},
                        {'soap': True, 'l_max': 4, 'n_max': 12, 'cutoff': 6,
                         'delta': 1, 'covariance_type': 'dot_product',
                         'zeta': 4, 'n_sparse': 100,
                         'sparse_method': 'cur_points',
                         'atom_gaussian_width': 0.3, 'add_species': False,
                         'n_species': 3, 'Z': 8, 'species_Z': [8, 1, 6]}]}

    gap_fit_string = gap_simple.dict_to_gap_fit_string(gap_fit_dict)

    expected_gap_fit_string = 'atoms_filename=train.xyz ' \
                              'default_sigma={0.01 0.1 0.1 0.0} sparse_seprate_file=F ' \
                              'core_ip_args="IP Glue" core_param_file=/test/path/test/file.xml ' \
                              'config_type_sigma=isolated_atom:1e-05:0.0:0.0:0.0:' \
                              'funky_configs:0.1:0.3:0.0:0.0 ' \
                              'gap={ soap=T l_max=6 n_max=12 cutoff=3 delta=1 ' \
                              'covariance_type=dot_product zeta=4 n_sparse=200 ' \
                              'sparse_method=cur_points : soap=T l_max=4 n_max=12 cutoff=6 delta=1' \
                              ' covariance_type=dot_product zeta=4 n_sparse=100 ' \
                              'sparse_method=cur_points atom_gaussian_width=0.3 add_species=F ' \
                              'n_species=3 Z=8 species_Z={{8 1 6}} }'

    assert gap_fit_string == expected_gap_fit_string

