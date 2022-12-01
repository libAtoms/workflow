import click
from wfl.configset import ConfigSet, OutputSpec
from ase.io.extxyz import key_val_str_to_dict


def _to_ConfigSet(ctx, param, value):
    return ConfigSet(value)

def inputs(f):
    """Add a standard option for the ConfigSet inputs"""
    f = click.option("--inputs", "-i", required=True, multiple=True, callback=_to_ConfigSet,
                    help='Input xyz file(s) to create ConfigSet from')(f) 
    return f


def _to_OutputSpec(ctx, param, value):
    return OutputSpec(value)


def outputs(f):
    """Add a standard option for the OutputSpec outputs"""
    f = click.option('--outputs', '-o', required=True, callback=_to_OutputSpec,
                     help="Ouput file to create OutputSpec from.")(f)
    return f

def _parse_extra_info(ctx, param, value):
    if value is not None:
        return key_val_str_to_dict(value)
    else:
        return {}

def extra_info(f):
    """Parse key=val string and return a dictionary"""
    f = click.option("--extra-info", "-i", callback=_parse_extra_info, help="Extra key=val pairs to add to Atoms.info")(f)
    return f

def param_fname(f):
    f = click.option("--param-fname", type=click.Path(), help="Path to the potential parameter file")(f)
    return f

def prop_prefix(f):
    f = click.option("--prop-prefix", "-p", help='Prefix to be pre-pended to all evaluate properties. Defaults to "gap_"/"ace_"/"mace_" as appropriate')(f)
    return f


def num_inputs_per_python_subprocess(f):
    f = click.option('--num-inputs-per-python-subprocess', default=10, 
    show_default=True, type=click.INT, 
    help="Number of configs to be evaluated per each calculator initialization")(f)
    return f

# def calc_property_prefix(f):
#     f = click.option('--calc-property-prefix', '-cpp', required=True, 
#     help="prefix for calculated (predicted) properties (energy, forces, ...)")(f)
#     return f

# def ref_property_prefix(f):
#     f = click.option("--ref-property-prefix", '-rpp', required=True, 
#     help="prefix for properties taken as reference (e.g. from electronic strucutre codes)")(f)
#     return f

# def config_properties(f):
#     f = click.option("--config-properties", '-cp', multiple=True, 
#     help="Multiple ``Atoms.info`` calculated properties (to be prefixed by ``ref_property_prefix`` or ``calc_property_prefix``) to"\
#          " compute error for.  ``virial`` will be reconstructed from ``stress``. Properties can end with ``/atom`` or "\
#          " ``/comp`` for different components being counted separately. Default of [\"energy/atom\", \"virial/atom/comp\"]only used if neither ``config_properties``"\
#          " nor ``atom_properties`` is present.")(f) 
#     return f

# def atom_properties(f):
#     f = click.option("--atom-properties", '-ap', multiple=True, 
#     help="Multiple `Atoms.arrays` calculated properties (to be prefixed by ``ref_property_prefix``"\
#          "or ``calc_property_prefix``) to compute error For.  Properties can end with ``/comp``"\
#          "for different components being computed separately, and ``/Z`` for different atomic numbers"\
#          "to be assigned to different categories.  Default of [\"forces\"] only used if neither ``config_properties``"\
#          "nor ``atom_properties`` is present.")(f) 
#     return f

# def category_keys(f):
#     f = click.option("--category-keys", '-ck', multiple=True, default=["config_type"], show_default=True,
#     help="Results will be averaged by a category, defined by a string containing the values of these"\
#          "keys in Atoms.info, in addition to overall average _ALL_ category.")(f)
#     return f

# def weight_property(f):
#     f = click.option("--weight-property", '-wp', help="Atoms.info key for weights to apply to RMSE calculation")(f)
#     return f

# def precision(f):
#     f = click.option("--precision", '-p', type=click.INT, default=2, help="precision for printing table")(f)
#     return f


