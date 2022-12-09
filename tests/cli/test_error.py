import os
import warnings
from click.testing import CliRunner
from wfl.cli.cli import cli


def test_error_table(tmp_path):
    """just makes sure code runs without error""" 

    ats_filename = os.path.join(os.path.abspath(os.path.dirname(__file__)),  '../assets', 'configs_for_error_test.xyz')
    fig_name = tmp_path / "error_scatter.png"
    warnings.warn(f"error plots in {fig_name}")


    command = "-v error "\
              f"--inputs {ats_filename} "\
              "--calc-property-prefix mace_ "\
              "--ref-property-prefix dft_ "\
              "--config-properties energy/atom "\
              "--config-properties energy "\
              "--atom-properties forces/comp/Z "\
              "--atom-properties forces/comp "\
              "--category-keys mol_or_rad "\
              f"--fig-name {fig_name} "


    print(command)

    runner = CliRunner()
    result = runner.invoke(cli, command)
    assert result.exit_code == 0
