import click

from quippy.potential import Potential

from wfl.autoparallelize.autoparainfo import AutoparaInfo
from wfl.cli import cli_options as opt
from wfl.calculators import generic


@click.command("gap")
@click.pass_context
@opt.inputs
@opt.outputs
@opt.param_fname
@opt.prop_prefix
@opt.num_inputs_per_python_subprocess
def gap(ctx, inputs, outputs, param_fname, prop_prefix, num_inputs_per_python_subprocess):
    """evaluates GAP"""
    
    if prop_prefix is None:
        prop_prefix="gap_"

    calc = (Potential, [], {"param_filename":param_fname})

    generic.run(
        inputs=inputs, 
        outputs=outputs,
        calculator=calc,
        output_prefix=prop_prefix, 
        autopara_info=AutoparaInfo(num_inputs_per_python_subprocess = num_inputs_per_python_subprocess))


def pyjulip_ace(param_fname):
    import pyjulip
    return pyjulip.ACE1(param_fname)


@click.command("ace")
@click.pass_context
@opt.inputs
@opt.outputs
@opt.param_fname
@opt.prop_prefix
@opt.num_inputs_per_python_subprocess
def ace(ctx, inputs, outputs, param_fname, prop_prefix, num_inputs_per_python_subprocess):
    """evaluates ACE"""

    if prop_prefix is None:
        prop_prefix = 'ace_'

    calc = (pyjulip_ace, [param_fname], {})

    generic.run(
        inputs=inputs, 
        outputs=outputs,
        calculator=calc,
        output_prefix=prop_prefix, 
        autopara_info=AutoparaInfo(num_inputs_per_python_subprocess = num_inputs_per_python_subprocess))


@click.command("mace")
@click.option("--dtype", default="float64", type=click.Choice(["float64", "float32"]), show_default=True, help="dtype MACE model was fitted with")
@click.pass_context
@opt.inputs
@opt.outputs
@opt.param_fname
@opt.prop_prefix
@opt.num_inputs_per_python_subprocess
def mace(ctx, inputs, outputs, param_fname, prop_prefix, num_inputs_per_python_subprocess, dtype):
    """evaluates MACE"""

    from mace.calculators.mace import MACECalculator 

    calc = (MACECalculator, [], {"model_path":param_fname, "default_dtype":dtype, "device":'cpu'})

    generic.run(
        inputs=inputs, 
        outputs=outputs,
        calculator=calc,
        output_prefix=prop_prefix, 
        autopara_info=AutoparaInfo(num_inputs_per_python_subprocess = num_inputs_per_python_subprocess))

