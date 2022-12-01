import os
from click.testing import CliRunner
from wfl.cli.cli import cli


def test_error_table(tmp_path):
    """just makes sure code runs without error""" 

    filename = os.path.join(os.path.abspath(os.path.dirname(__file__)),  '../assets', 'configs_for_error_test.xyz')

    command = "-v error table "\
              f"--inputs {filename} "\
              "--calc-property-prefix mace_ "\
              "--ref-property-prefix dft_ "\
              "--config-properties energy/atom "\
              "--config-properties energy "\
              "--atom-properties forces/comp/Z "\
              "--atom-properties forces/comp "\
              "--category-keys mol_or_rad "  

    print(command)

    runner = CliRunner()
    result = runner.invoke(cli, command)
    assert result.exit_code == 0
