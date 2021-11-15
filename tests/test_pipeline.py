import pytest

from wfl.configset import ConfigSet_out
from wfl.generate_configs import buildcell


def test_empty_iterator(tmp_path):
    co = buildcell.run(ConfigSet_out(output_files=str(tmp_path / 'dummy.xyz')), range(0), buildcell_cmd='dummy', buildcell_input='dummy')

    assert len([at for at in co]) == 0
