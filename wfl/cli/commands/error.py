import click
import pandas as pd
from wfl.cli import cli_options as opt
from wfl.fit.error import calc as ref_err_calc
from wfl.fit.error import select_units

@click.command("table")
@click.option('--calc-property-prefix', '-cpp', required=True, 
    help="prefix for calculated (predicted) properties (energy, forces, ...)")
@click.option("--ref-property-prefix", '-rpp', required=True, 
    help="prefix for properties taken as reference (e.g. from electronic strucutre codes)")
@click.option("--config-properties", '-cp', multiple=True, 
    help="Multiple ``Atoms.info`` calculated properties (to be prefixed by ``ref_property_prefix`` or ``calc_property_prefix``) to"\
         " compute error for.  ``virial`` will be reconstructed from ``stress``. Properties can end with ``/atom`` or "\
         " ``/comp`` for different components being counted separately. Default of [\"energy/atom\", \"virial/atom/comp\"]only used if neither ``config_properties``"\
         " nor ``atom_properties`` is present.")
@click.option("--atom-properties", '-ap', multiple=True, 
    help="Multiple `Atoms.arrays` calculated properties (to be prefixed by ``ref_property_prefix``"\
         "or ``calc_property_prefix``) to compute error For.  Properties can end with ``/comp``"\
         "for different components being computed separately, and ``/Z`` for different atomic numbers"\
         "to be assigned to different categories.  Default of [\"forces\"] only used if neither ``config_properties``"\
         "nor ``atom_properties`` is present.")
@click.option("--category-keys", '-ck', multiple=True, default=["config_type"], show_default=True,
    help="Results will be averaged by a category, defined by a string containing the values of these"\
         "keys in Atoms.info, in addition to overall average _ALL_ category.")
@click.option("--weight-property", '-wp', help="Atoms.info key for weights to apply to RMSE calculation")
@click.option("--precision", '-p', type=click.INT, default=2, help="precision for printing table")
@click.pass_context
@opt.inputs
def table(ctx, inputs, calc_property_prefix, ref_property_prefix, 
          config_properties, atom_properties, category_keys,
          weight_property, precision):
        """Prints error summary table"""
        # TODO
        # - atomization energy
        # - multiple files for intpus - for isolated atoms and not
        # - rename "num" to "count"

        errors, diffs, parity = ref_err_calc(
            inputs=inputs, 
            calc_property_prefix=calc_property_prefix,
            ref_property_prefix=ref_property_prefix,
            config_properties=config_properties,
            atom_properties=atom_properties,
            category_keys=category_keys,
            weight_property=weight_property)

        df = format_errors(errors)
        pd.set_option('display.max_rows', None)
        pd.set_option('display.max_columns', None)
        pd.set_option('display.colheader_justify', 'center')
        pd.set_option('display.precision', precision)
        print(df)

def format_errors(errors):
    # make dataframe in the correct orientation 
    df = pd.DataFrame.from_dict(errors, orient="index").stack()
    df = pd.json_normalize(df).set_index(df.index)

    # change and label units
    df["MAE"] = df["MAE"].apply(lambda x: x*1e3)
    df["RMSE"] = df["RMSE"].apply(lambda x: x*1e3)
    df["units"] = [select_units(prop, "error") for prop, _ in df.index]

    return df



@click.command("scatter")
def scatter():
    pass