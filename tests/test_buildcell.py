import os
import json
import shutil

import pytest

from wfl.configset import OutputSpec
from wfl.generate import buildcell

def test_buildcell(tmp_path):

    do_buildcell(tmp_path, 'dummy.xyz')


@pytest.mark.remote
def test_buildcell_remote(tmp_path, expyre_systems, monkeypatch, remoteinfo_env):
    for sys_name in expyre_systems:
        if sys_name.startswith('_'):
            continue

        do_buildcell_remote(tmp_path, sys_name, monkeypatch, remoteinfo_env)


def do_buildcell_remote(tmp_path, sys_name, monkeypatch, remoteinfo_env):
    ri = {'sys_name': sys_name, 'job_name': 'pytest_'+sys_name,
          'resources': {'max_time': '1h', 'num_nodes': 1},
          'num_inputs_per_queued_job': -36, 'check_interval': 10}

    remoteinfo_env(ri)

    do_buildcell(tmp_path, f'dummy_{sys_name}.xyz')


def do_buildcell(tmp_path, filename):

    if 'WFL_PYTEST_BUILDCELL' in os.environ:
        buildcell_cmd = os.environ['WFL_PYTEST_BUILDCELL']
    elif shutil.which("buildcell"):
        buildcell_cmd = shutil.which("buildcell")
    else:
        pytest.skip('buildcell tests need WFL_PYTEST_BUILDCELL with path to buildcell executable or having it locatable via PATH')

    buildcell_input="""#TARGVOL=19-21
#SPECIES=Li%NUM=1
#NFORM={6,8,10,12,14,16,18,20,22,24}
#SYMMOPS=1-8
#SLACK=0.25
#OVERLAP=0.1
#COMPACT
#MINSEP=0.5 Li-Li=2.7
##EXTRA_INFO RSS_min_vol_per_atom=10.0"""

    co = buildcell.buildcell(range(100), OutputSpec(tmp_path / filename),
                       buildcell_cmd=buildcell_cmd, buildcell_input=buildcell_input)

    assert len(list(co)) == 100


@pytest.mark.parametrize(
    ('inputs', 'expected_output_values'),
    (
        pytest.param(
            (3, 5, 2, None, 0.5, (0.95, 1.05), 0.9, '1-8', (6, 24), None),
            ('4.8-5.2', 'Li%NUM=1', '{6,8,10,12,14,16,18,20,22,24}', '1-8', '0.25', '0.1', '0.5 Li-Li=1.8',
            'RSS_min_vol_per_atom=2.5'),
            id='single atomic species'
        ),
        pytest.param(
            ([3, 13], [5, 10], [2, 3], [1, 2], 0.5, (0.95, 1.05), 0.9, '1-8', (6, 24), None),
            ('16-18', 'Li%NUM=1,Al%NUM=2', '{2,4,6,8}', '1-8', '0.25', '0.1', '0.5 Li-Li=1.8 Li-Al=2.2 Al-Al=2.7',
             'RSS_min_vol_per_atom=4.166666666666667'),
            id='multiple atomic species'
        ),
        pytest.param(
            (3, 5, 2, None, 0.25, (0.95, 1.05), 0.9, '1-8', (6, 24), None),
            ('4.8-5.2', 'Li%NUM=1', '{6,8,10,12,14,16,18,20,22,24}', '1-8', '0.25', '0.1', '0.5 Li-Li=1.8',
             'RSS_min_vol_per_atom=1.25'),
            id='RSS_min_vol_factor'
        ),
        pytest.param(
            (3, 5, 2, None, 0.5, (0.90, 1.11), 0.9, '1-8', (6, 24), None),
            ('4.5-5.6', 'Li%NUM=1', '{6,8,10,12,14,16,18,20,22,24}', '1-8', '0.25', '0.1', '0.5 Li-Li=1.8',
             'RSS_min_vol_per_atom=2.5'),
            id='vol_range'
        ),
        pytest.param(
            (3, 5, 2, None, 0.5, (0.95, 1.05), 0.8, '1-8', (6, 24), None),
            ('4.8-5.2', 'Li%NUM=1', '{6,8,10,12,14,16,18,20,22,24}', '1-8', '0.25', '0.1', '0.5 Li-Li=1.6',
             'RSS_min_vol_per_atom=2.5'),
            id='min_sep_fac'
        ),
        pytest.param(
            (3, 5, 2, None, 0.5, (0.95, 1.05), 0.9, '1-24', (6, 24), None),
            ('4.8-5.2', 'Li%NUM=1', '{6,8,10,12,14,16,18,20,22,24}', '1-24', '0.25', '0.1', '0.5 Li-Li=1.8',
             'RSS_min_vol_per_atom=2.5'),
            id='symmops'
        ),
        pytest.param(
            (3, 5, 2, None, 0.5, (0.95, 1.05), 0.9, '1-8', (2, 5), None),
            ('4.8-5.2', 'Li%NUM=1', '{2,4}', '1-8', '0.25', '0.1', '0.5 Li-Li=1.8',
             'RSS_min_vol_per_atom=2.5'),
            id='natoms'
        ),
        pytest.param(
            (3, 5, 2, None, 0.5, (0.95, 1.05), 0.9, '1-8', (2, 5), 'only'),
            ('4.8-5.2', 'Li%NUM=1', '{3,5}', '1-8', '0.25', '0.1', '0.5 Li-Li=1.8',
             'RSS_min_vol_per_atom=2.5'),
            id='odd="only"'
        ),
        pytest.param(
            (3, 5, 2, None, 0.5, (0.95, 1.05), 0.9, '1-8', (2, 5), 'also'),
            ('4.8-5.2', 'Li%NUM=1', '{2,3,4,5}', '1-8', '0.25', '0.1', '0.5 Li-Li=1.8',
             'RSS_min_vol_per_atom=2.5'),
            id='odd="also"'
        ),
    )
)
def test_create_input(inputs, expected_output_values):
    output_templates = [
        '#TARGVOL={}', '#SPECIES={}', '#NFORM={}', '#SYMMOPS={}', '#SLACK={}', '#OVERLAP={}', '#MINSEP={}',
        '##EXTRA_INFO {}',
        ]
    output = buildcell.create_input(*inputs)
    for temp_i, val_i in zip(output_templates, expected_output_values):
        assert temp_i.format(val_i) in output
