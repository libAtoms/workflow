import os
import shutil
import json
import pytest
from expyre import config 

from .calculators.test_aims import aims_prerequisites


def _get_coding_blocks(nb_file):
    """Parse ```nb_file``` for coding blocks and return as list of strings."""
    with open(nb_file, 'r') as fo:
        nb = json.load(fo)
    return [''.join(cell['source']) for cell in nb['cells'] if cell['cell_type'] == 'code']


@pytest.mark.parametrize(
    ('nb_file', 'idx_execute', 'needs_expyre'),
    (
        pytest.param('examples.buildcell.ipynb', 'all', False, id='buildcell',
            marks=pytest.mark.skipif(not shutil.which("buildcell"), reason="buildcell not in PATH")),
        pytest.param('examples.dimers.ipynb', 'all', False, id='dimer structures'),
        pytest.param('examples.select_fps.ipynb', 'all', False, id='select fps'),
        pytest.param('examples.fhiaims_calculator.ipynb', 'all', False, id='fhiaims_calculator',
            marks=aims_prerequisites),
        pytest.param("examples.daisy_chain_mlip_fitting.ipynb", "all", True, id="daisy_chain_mlip_fitting")
    )
)
def test_example(tmp_path, nb_file, idx_execute, monkeypatch, needs_expyre, expyre_systems):
    if needs_expyre and "github" not in expyre_systems:
        pytest.skip(reason=f'Notebook {nb_file} requires ExPyRe, but system "github" is not in config.json or'
              f'"EXPYRE_PYTEST_SYSTEMS" environment variable.')
    print("running test_example", nb_file)
    basepath = os.path.join(f'{os.path.dirname(__file__)}/../docs/source')
    coding_blocks = _get_coding_blocks(f'{basepath}/{nb_file}')
    code = '\n'.join([cb_i for idx_i, cb_i in enumerate(coding_blocks) if idx_execute == 'all' or idx_i in idx_execute])
    assert code is not ''

    monkeypatch.chdir(tmp_path)
    try:
        exec(code, globals())
    except Exception as exc:
        import traceback, re
        tb_str = traceback.format_exc()
        line_nos = list(re.findall("line ([0-9]+),", tb_str))
        line_no = int(line_nos[-1])
        lines = list(enumerate(code.splitlines()))[line_no - 5 : line_no + 5]
        actual_error = "\n".join([f"{li:4d}{'*' if li == line_no else ' '} {l}" for li, l in lines])

        raise RuntimeError(f"Exception raised by test_example {nb_file}, traceback:\n{actual_error}\n") from exc
