import os
import shutil
import json
import pytest

from .calculators.test_aims import aims_prerequisites


def _get_coding_blocks(nb_file):
    """Parse ```nb_file``` for coding blocks and return as list of strings."""
    with open(nb_file, 'r') as fo:
        nb = json.load(fo)
    return [''.join(cell['source']) for cell in nb['cells'] if cell['cell_type'] == 'code']


@pytest.mark.parametrize(
    ('nb_file', 'idx_execute'),
    (
        pytest.param('examples.buildcell.ipynb', 'all', id='buildcell',
            marks=pytest.mark.skipif(not shutil.which("buildcell"), reason="buildcell not in PATH")),
        pytest.param('examples.dimers.ipynb', 'all', id='dimer structures'),
        pytest.param('examples.select_fps.ipynb', 'all', id='select fps'),
        pytest.param('examples.fhiaims_calculator.ipynb', 'all', id='fhiaims_calculator',
            marks=aims_prerequisites),
    )
)
def test_example(tmp_path, nb_file, idx_execute):
    basepath = os.path.join(f'{os.path.dirname(__file__)}/../docs/source')
    coding_blocks = _get_coding_blocks(f'{basepath}/{nb_file}')
    code = '\n'.join([cb_i for idx_i, cb_i in enumerate(coding_blocks) if idx_execute == 'all' or idx_i in idx_execute])
    assert code is not ''

    os.chdir(tmp_path)
    exec(code)
    os.chdir('..')
