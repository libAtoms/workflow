import click
from wfl.cli import cli_options as opt


@click.command("smiles")
@click.argument("smiles-string", nargs=-1)
@click.pass_context
@opt.outputs
@opt.extra_info
def smiles(ctx, smiles_string, extra_info, outputs):
    """Generate xyz from SMILES strings"""

    import wfl.generate.smiles

    verbose = ctx.obj["verbose"]

    if verbose:
        print(f'smiles: {smiles_string}')
        print(f'info: {extra_info}')
        print(outputs)

    wfl.generate.smiles.smiles(smiles_string, outputs=outputs, extra_info=extra_info)


@click.command("buildcell")
@click.option('--buildcell-input', required=True, help='buildcell input file')
@click.option("--buildcell-exec", required=True, help="buildcell executable including path")
@click.option("--n-configs", "-N", type=click.INT, required=True, help="number of configs to generate")
@click.option("--perturbation", type=click.FLOAT, default=0.0, help="magnitude of random perturbation to atomic positions")
@click.pass_context
@opt.outputs
@opt.extra_info
def buildcell(ctx, outputs, buildcell_input, buildcell_exec, n_configs,
              extra_info, perturbation):
    """Repeatedly runs buildcell (from Pickard's AIRSS distribution) to generate random configs with
    specified species, volumes, distances, symmetries, etc.

    Minimal contents of --buildcell-input file:

    \b
    #TARGVOL=<min_vol>-<max_vol> (NOTE: volume is volume_per_formula_unit/number_of_species)
    #SPECIES=<elem_symbol_1>%NUM=<num_1>[,<elem_symbol_2>%NUM=<num_2 ...]
    #NFORM=[ <n_min>-<n_max> | { <n_1>, <n_2>, ... } ]
    #SYMMOPS=<n_min>-<n_max> (NOTE: optional)
    #SLACK=0.25
    #OVERLAP=0.1
    #COMPACT
    #MINSEP=<min_separation_default> <elem_symbol_1>-<elem_symbol_1>=<min_separation_1_1> [
    <elem_symbol_1>-<elem_symbol_2=<min_separation_1_2> ... ]
    ##EXTRA_INFO <info_field>=<value> (NOTE: optional)
    """
    import wfl.generate.buildcell

    with open(buildcell_input) as bc_f:
        buildcell_input_txt = bc_f.read()

    wfl.generate.buildcell.buildcell(
        outputs=outputs,
        inputs=range(n_configs),
        buildcell_cmd=buildcell_exec,
        buildcell_input=buildcell_input_txt,
        extra_info=extra_info,
        perturbation=perturbation,
        verbose=ctx.obj["verbose"])
