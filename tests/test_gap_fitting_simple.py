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
                       'config_type_sigma': 'cfg1:1.0:2.0:0.3:4.0:cfg2:1:1:1:1',
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


@pytest.mark.skipif(not shutil.which("gap_fit"), reason="gap_fit not in PATH")  # skips it if gap_fit not in path
def test_fitting_gap_cli(quippy, tmp_path):
    runner = CliRunner()
    assets_dir = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'assets')
    param_filename = os.path.join(assets_dir, 'simple_gap_fit_parameters.yml')
    gap_train_fname = os.path.join(assets_dir, 'simple_gap_training_set.xyz')

    gap_file = os.path.join(tmp_path, 'gap_test.xml')

    # functions wrapped in click
    command = ["fitting", "simple-gap", "--atoms-filename", gap_train_fname, "-g", gap_file, "-p", param_filename, "--fit" ]
    result = runner.invoke(cli, command)
    if result.exit_code != 0:
        print('OUTPUT')
        print(result.output)
        print('Exception')
        print(result.exception)
        print('Exception info')
        print(result.exc_info)
        import traceback
        traceback.print_exception(*result.exc_info)
    assert result.exit_code == 0

    print('only test is checking for existence of', gap_file, 'and related')
    assert os.path.isfile(gap_file)
    assert os.path.isfile(gap_file.replace('.xml', '_output.txt'))


@pytest.mark.skipif(not shutil.which("gap_fit"), reason="gap_fit not in PATH")  # skips it if gap_fit not in path
@pytest.mark.remote
def test_fitting_gap_cli_remote(quippy, tmp_path, expyre_systems, monkeypatch):
    mypath = Path(__file__).parent.parent
    ri = {'resources' : {'max_time': '10m', 'n': [1, 'nodes']}}

    for sys_name in expyre_systems:
        if sys_name.startswith('_'):
            continue

        ri['sys_name'] = sys_name
        ri['job_name'] = 'pytest_gap_fit_'+sys_name

        if 'WFL_PYTEST_REMOTEINFO' in os.environ:
            ri_extra = json.loads(os.environ['WFL_PYTEST_REMOTEINFO'])
            if 'resources' in ri_extra:
                ri['resources'].update(ri_extra['resources'])
                del ri_extra['resources']
            ri.update(ri_extra)

        monkeypatch.setenv('WFL_GAP_SIMPLE_FIT_REMOTEINFO', json.dumps(ri))
        test_fitting_gap_cli(quippy, tmp_path)
