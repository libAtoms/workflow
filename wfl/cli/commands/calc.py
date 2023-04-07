import click

from quippy.potential import Potential

from wfl.autoparallelize.autoparainfo import AutoparaInfo
from wfl.cli import cli_options as opt
from wfl.calculators import generic
from wfl.utils import configs
import wfl.descriptors.quippy


@click.command("gap")
@click.pass_context
@opt.inputs
@opt.outputs
@opt.param_fname
@opt.prop_prefix
@opt.num_inputs_per_python_subprocess
def gap(ctx, inputs, outputs, param_fname, prop_prefix, num_inputs_per_python_subprocess):
    """evaluate GAP"""
    
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
    """evaluate ACE"""

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
    """evaluate MACE"""

    from mace.calculators.mace import MACECalculator 

    calc = (MACECalculator, [], {"model_path":param_fname, "default_dtype":dtype, "device":'cpu'})

    generic.run(
        inputs=inputs, 
        outputs=outputs,
        calculator=calc,
        output_prefix=prop_prefix, 
        autopara_info=AutoparaInfo(num_inputs_per_python_subprocess = num_inputs_per_python_subprocess))


@click.command("atomization-energy")
@click.pass_context
@opt.inputs
@opt.outputs
@opt.prop_prefix
@click.option("--prop", default="energy", show_default=True, 
    help="Property to calculate atomization value for")
@click.option("--isolated-atom-info-key", "-k", default="config_type", show_default=True, 
    help="``atoms.info`` key on which to select isolated atoms")
@click.option("--isolated-atom-info-value", "-v", default="default", 
    help="``atoms.info['isolated_atom_info_key']`` value for isolated atoms. Defaults to \"IsolatedAtom\" or \"isolated_atom\"")
def atomization_energy(ctx, inputs, outputs, prop_prefix, prop, isolated_atom_info_key, isolated_atom_info_value):
    """Calculate atomization energy"""
    configs.atomization_energy(
        inputs=inputs,
        outputs=outputs,
        prop_prefix=prop_prefix,
        property=prop,
        isolated_atom_info_key=isolated_atom_info_key,
        isolated_atom_info_value=isolated_atom_info_value
    )


