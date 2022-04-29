import os
import json

import pytest

from wfl.configset import OutputSpec
from wfl.generate import buildcell


def test_buildcell(tmp_path):

    do_buildcell(tmp_path, 'dummy.xyz')


@pytest.mark.remote
def test_buildcell_remote(tmp_path, expyre_systems, monkeypatch):
    for sys_name in expyre_systems:
        if sys_name.startswith('_'):
            continue

        do_buildcell_remote(tmp_path, sys_name, monkeypatch)


def do_buildcell_remote(tmp_path, sys_name, monkeypatch):
    ri = {'sys_name': sys_name, 'job_name': 'test_'+sys_name,
          'resources': {'max_time': '1h', 'n': (1, 'nodes')},
          'job_chunksize': -36, 'check_interval': 10}

    if 'WFL_PYTEST_REMOTEINFO' in os.environ:
        ri_extra = json.loads(os.environ['WFL_PYTEST_REMOTEINFO'])
        if 'resources' in ri_extra:
            ri['resources'].update(ri_extra['resources'])
            del ri_extra['resources']
        ri.update(ri_extra)

    do_buildcell(tmp_path, f'dummy_{sys_name}.xyz')


def do_buildcell(tmp_path, filename):
    if 'WFL_PYTEST_BUILDCELL' not in os.environ:
        pytest.skip('buildcell tests need WFL_PYTEST_BUILDCELL with path to buildcell executable')

    buildcell_input="""#TARGVOL=19-21
#SPECIES=Li%NUM=1
#NFORM={6,8,10,12,14,16,18,20,22,24}
#SYMMOPS=1-8
#SLACK=0.25
#OVERLAP=0.1
#COMPACT
#MINSEP=0.5 Li-Li=2.7
##EXTRA_INFO RSS_min_vol_per_atom=10.0"""

    co = buildcell.run(OutputSpec(output_files=str(tmp_path / filename)), range(100),
                      buildcell_cmd=os.environ['WFL_PYTEST_BUILDCELL'], buildcell_input=buildcell_input)

    assert len(list(co)) == 100
