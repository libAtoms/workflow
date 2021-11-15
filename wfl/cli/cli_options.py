"""
These are generalised cli options, merged decorators for repeated operations
"""

from click import STRING, argument, option


def file_input_options(f):
    """"""
    f = argument("inputs", nargs=-1)(f)
    f = option("--index", "-i", type=STRING, required=False, help="Pass this index to configset globally")(f)
    return f
