import click
from wfl.configset import ConfigSet, OutputSpec
from ase.io.extxyz import key_val_str_to_dict


def _to_ConfigSet(ctx, param, value):
    return ConfigSet(value)

def inputs(f):
    """Create ConfigSet for given filename(s)"""
    f = click.option("--inputs", "-i", required=True, multiple=True, callback=_to_ConfigSet,
                    help='Input xyz file(s) to create ConfigSet from')(f) 
    return f


def _to_OutputSpec(ctx, param, value):
    return OutputSpec(value)


def outputs(f):
    """Create OutputSpec for given filename"""
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
