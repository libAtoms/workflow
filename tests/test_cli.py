import os
import click
from click.testing import CliRunner
import pytest
from ase import Atoms
from ase.io import write
from wfl.cli.cli import cli

@pytest.mark.skipif("ASE_ORCA_COMMAND" not in os.environ, reason="no ORCA executable in path")
def test_orca_eval(cli_runner, tmp_path):

    atoms = Atoms("H2", positions=[(0, 0, 0), (0, 0, 0.9)])
    atoms = [atoms] * 3 

    fn_in = tmp_path / "ats_in.xyz"
    fn_out = tmp_path / "ats_out.xyz"

    write(fn_in, atoms)
    workdir = tmp_path / "workdir"

    params = [
        '-v',
        "ref-method",
        "orca-eval",
        f"--output-file {str(fn_out)}",
        f"--output-prefix orca_",
        f"--workdir {workdir}", 
        str(fn_in)
        ]

    runner = CliRunner()
    result = runner.invoke(cli, " ".join(params))

    assert fn_out.exists()