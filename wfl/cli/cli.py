import warnings

import click


@click.group("wfl")
@click.option("--verbose", "-v", is_flag=True)
@click.pass_context
def cli(ctx, verbose):
    """Workflow command line interface.
    """
    ctx.ensure_object(dict)
    ctx.obj["verbose"] = verbose

    # ignore calculator writing warnings
    if not verbose:
        warnings.filterwarnings("ignore", category=UserWarning, module="ase.io.extxyz")


from wfl.cli.commands.error import show_error
cli.add_command(show_error)

@cli.group("generate")
@click.pass_context
def subcli_generate(ctx):
    """Generate structures"""
    pass

from wfl.cli.commands.generate import smiles, buildcell
subcli_generate.add_command(smiles)
subcli_generate.add_command(buildcell)


@cli.group("select")
@click.pass_context
def subcli_select(ctx):
    """Select structures from database"""
    pass

from wfl.cli.commands.select import cur, by_lambda
subcli_select.add_command(cur)
subcli_select.add_command(by_lambda)

@cli.group("calc")
@click.pass_context
def subcli_calc(ctx):
    "Calculate properties and descriptors."
    pass

from wfl.cli.commands.calc import gap, ace, mace, atomization_energy, quippy

subcli_calc.add_command(gap)
subcli_calc.add_command(ace)
subcli_calc.add_command(mace)
subcli_calc.add_command(atomization_energy)
subcli_calc.add_command(quippy)





