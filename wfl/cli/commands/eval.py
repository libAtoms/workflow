import click

from wfl.autoparallelize import AutoparaInfo
from wfl.cli import cli_options as opt
from wfl.calculators import generic
from wfl.utils import configs


def pyjulip_ace(param_fname):
    import pyjulip
    return pyjulip.ACE1(param_fname)


@click.command("gap")
@click.pass_context
@opt.inputs
@opt.outputs
@opt.param_fname
@opt.kwargs
@opt.prop_prefix
@opt.num_inputs_per_python_subprocess
def gap(ctx, inputs, outputs, param_fname, kwargs, prop_prefix, num_inputs_per_python_subprocess):
    """evaluates GAP"""

    from quippy.potential import Potential

    if prop_prefix is None:
        prop_prefix = "gap_"

    kwargs_use = {"param_filename": param_fname}
    kwargs_use.update(kwargs)
    calc = (Potential, [], kwargs_use)

    generic.calculate(
        inputs=inputs,
        outputs=outputs,
        calculator=calc,
        output_prefix=prop_prefix,
        autopara_info=AutoparaInfo(num_inputs_per_python_subprocess=num_inputs_per_python_subprocess))


@click.command("ace")
@click.pass_context
@opt.inputs
@opt.outputs
@opt.param_fname
@opt.kwargs
@opt.prop_prefix
@opt.num_inputs_per_python_subprocess
def ace(ctx, inputs, outputs, param_fname, kwargs, prop_prefix, num_inputs_per_python_subprocess):
    """evaluates ACE"""

    if prop_prefix is None:
        prop_prefix = 'ace_'

    calc = (pyjulip_ace, [param_fname], kwargs)

    generic.calculate(
        inputs=inputs,
        outputs=outputs,
        calculator=calc,
        output_prefix=prop_prefix,
        autopara_info=AutoparaInfo(num_inputs_per_python_subprocess=num_inputs_per_python_subprocess))


@click.command("mace")
@click.pass_context
@opt.inputs
@opt.outputs
@opt.param_fname
@opt.kwargs
@opt.prop_prefix
@opt.num_inputs_per_python_subprocess
def mace(ctx, inputs, outputs, param_fname, kwargs, prop_prefix, num_inputs_per_python_subprocess):
    """evaluates MACE"""

    from mace.calculators import MACECalculator

    if prop_prefix is None:
        prop_prefix = 'mace_'

    kwargs_use = {"model_paths": param_fname, "device": "cpu"}
    kwargs_use.update(kwargs)
    calc = (MACECalculator, [], kwargs_use)

    generic.calculate(
        inputs=inputs,
        outputs=outputs,
        calculator=calc,
        output_prefix=prop_prefix,
        autopara_info=AutoparaInfo(num_inputs_per_python_subprocess=num_inputs_per_python_subprocess))


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
def atomization_energy(inputs, outputs, prop_prefix, prop, isolated_atom_info_key, isolated_atom_info_value):
    configs.atomization_energy(
        inputs=inputs,
        outputs=outputs,
        prop_prefix=prop_prefix,
        property=prop,
        isolated_atom_info_key=isolated_atom_info_key,
        isolated_atom_info_value=isolated_atom_info_value)
