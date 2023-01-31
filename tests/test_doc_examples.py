import os
import shutil
import re
from pathlib import Path
import json
import pytest


def _get_coding_blocks(nb_file):
    """Parse ```nb_file``` for coding blocks and return as list of strings."""
    with open(nb_file, 'r') as fo:
        nb = json.load(fo)
    return [''.join(cell['source']) for cell in nb['cells'] if cell['cell_type'] == 'code']

def _fix_wfl(code_block):
    if re.match(r"!\s*wfl\s+", code_block):
        # join lines
        code_block = re.sub(r"\\\n", " ", code_block)
        # find args for cli runner
        args_str = re.sub(r"^!\s*wfl\s+", "", code_block)
        # run cli comand
        code_block = ("from click.testing import CliRunner\n" +
                      "from wfl.cli.cli import cli\n" +
                      "import shlex\n" +
                      "runner = CliRunner()\n" +
                      f"runner.invoke(cli, shlex.split(\"\"\"{args_str}\"\"\"))")

    return code_block

@pytest.mark.parametrize(
    ('nb_file', 'idx_execute', 'wfl_cli'),
    (
        pytest.param('examples.buildcell.ipynb', 'all', False, id='buildcell',
            marks=pytest.mark.skipif(not shutil.which("buildcell"), reason="buildcell not in PATH")),
<<<<<<< HEAD
        pytest.param('examples.dimers.ipynb', 'all', False, id='dimer structures'),
        pytest.param('examples.cli.ipynb', 'all', True, id='command line interface')
=======
        pytest.param('examples.dimers.ipynb', 'all', id='dimer structures'),
        pytest.param('examples.select_fps.ipynb', 'all', id='select fps')
>>>>>>> main
    )
)

def test_example(tmp_path, monkeypatch, nb_file, idx_execute, wfl_cli):
    basepath = os.path.join(f'{os.path.dirname(__file__)}/../docs/source')
    coding_blocks = _get_coding_blocks(f'{basepath}/{nb_file}')
    if wfl_cli:
        coding_blocks_exec = [_fix_wfl(cb_i) for idx_i, cb_i in enumerate(coding_blocks) if idx_execute == 'all' or idx_i in idx_execute]

        example_name = nb_file.replace("examples.", "").replace(".ipynb", "")

        source_example_files_dir = (Path(__file__).parent.parent / "docs" / "example_files" / example_name)
        if source_example_files_dir.is_dir():
            pytest_example_files_dir = tmp_path / "example_files" / example_name
            shutil.copytree(source_example_files_dir, pytest_example_files_dir)

            # examples look for their files under "../example_files/<example_name>/", so move
            # down one to allow the ".." to work
            (tmp_path / "run").mkdir()
            monkeypatch.chdir(tmp_path / "run")
    else:
        coding_blocks_exec = [cb_i for idx_i, cb_i in enumerate(coding_blocks) if idx_execute == 'all' or idx_i in idx_execute]

    code = '\n'.join(coding_blocks_exec)
    assert code is not ''

    exec(code)
