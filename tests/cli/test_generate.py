from click.testing import CliRunner
from ase.io import read
from wfl.cli.cli import cli

def test_generate_smiles(tmp_path):

    fn_out = tmp_path / "ats_out.xyz"

    params = [
        '-v', 
        'generate', 
        'smiles', 
        f'-o {str(fn_out)}', 
        '-i "config_type=rdkit" '
        'CCC'
    ]

    runner = CliRunner()
    result = runner.invoke(cli, " ".join(params))

    ats = read(fn_out)
    assert len(ats) ==11
    assert ats.info["config_type"] == "rdkit"


def test_generate_buildcell():
    pass

