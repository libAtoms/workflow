import click

from quippy.potential import Potential

from wfl.autoparallelize.autoparainfo import AutoparaInfo
from wfl.cli import cli_options as opt


@click.command("gap")
@click.option("--gap-fname", '-g', type=click.Path(), help="Path to GAP XML")
@click.option("--prop-prefix", '-p', default='gap_', show_default=True,
                help='Prefix to be pre-pended to all evaluate properties')
@click.option("--num-inputs-per-python-subprocess", default=10, show_default=True,
                type=click.INT, help="Number of configs to be evaluated per each calculator initialization")
@click.pass_context
@opt.inputs
@opt.outputs
def gap(inputs, outputs, gap_fname, prop_prefix, num_inputs_per_python_subprocess):
    """evaluates GAP"""

    calc = (Potential, [], {"param_filename":gap_fname})

    generic.run(
        inputs=inputs, 
        outputs=outputs,
        calculator=calc,
        output_prefix=prop_prefix, 
        autopara_info=AutoparaInfo(num_inputs_per_python_subprocess = num_inputs_per_python_subprocess))


def pyjulip_ace(ace_fname):
    import pyjulip
    return pyjulip.ACE1(ace_fname)


@click.command("ace")
@click.option("--ace-fname", '-a', type=click.Path(), help="Path to ACE JSON")
@click.option("--prop-prefix", '-p', default='ace_', show_default=True,
                help='Prefix to be pre-pended to all evaluate properties')
@click.option("--num-inputs-per-python-subprocess", default=10, show_default=True,
                type=click.INT, help="Number of configs to be evaluated per each calculator initialization")
@click.pass_context
@opt.inputs
@opt.outputs
def ace(inputs, outputs, ace_fname, prop_prefix, num_inputs_per_python_subprocess):
    """evaluates ACE"""

    calc = (pyjulip_ace, [ace_fname], {})

    generic.run(
        inputs=inputs, 
        outputs=outputs,
        calculator=calc,
        output_prefix=prop_prefix, 
        autopara_info=AutoparaInfo(num_inputs_per_python_subprocess = num_inputs_per_python_subprocess))


@click.command("mace")
@click.option("--mace-fname", '-m', type=click.Path(), help="Path to MACE filename")
@click.option("--prop-prefix", '-p', default='mace_', show_default=True,
                help='Prefix to be pre-pended to all evaluate properties')
@click.option("--num-inputs-per-python-subprocess", default=10, show_default=True,
                type=click.INT, help="Number of configs to be evaluated per each calculator initialization")
@click.option("--dtype", default="float64", type=click.Choice(["float64", "float32"]), show_default=True, help="dtype MACE model was fitted with")
@click.option("--device", default="cpu", show_default=True, type=click.Choice(["cpu", "cuda"]), help='model type')
@click.pass_context
@opt.inputs
@opt.outputs
def mace(inputs, outputs, mace_fname, prop_prefix, num_inputs_per_python_subprocess, dtype, device):
    """evaluates MACE"""

    from mace.calculators.mace import MACECalculator 

    calc = (MACECalculator, [], {"model_path":mace_fname, "default_dtype":dtype, "device":device})

    generic.run(
        inputs=inputs, 
        outputs=outputs,
        calculator=calc,
        output_prefix=prop_prefix, 
        autopara_info=AutoparaInfo(num_inputs_per_python_subprocess = num_inputs_per_python_subprocess))

