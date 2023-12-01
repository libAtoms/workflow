"""
QUIP-related string manipulations
"""
from ase.io.extxyz import key_val_dict_to_str


def dict_to_quip_str(d, list_brackets='{}'):
    """dictionary to QUIP CLI string

    Parameters
    ----------
    d: dict
        descriptor key-value pairs
    list_brackets: str, default '{}'
        string containing open and close symbols for lists (usually '{}' or '{{}}')

    Returns
    -------
    str: descriptor string
    """

    assert len(list_brackets) % 2 == 0

    def _list_join(sep, v):
        if isinstance(v, str):
            # strings are iterable but need to be used as is
            return v

        try:
            # try treating as an iterable
            return sep.join([str(vv) for vv in v])
        except TypeError:
            return v

    string = ''
    for key, val in d.items():
        if isinstance(val, list):
            # special treatment for lists, normally in brackets, and can be other things like
            # double brackets
            string += f'{key}=' + (list_brackets[0:len(list_brackets) // 2] +
                                   ' '.join([str(v) for v in val]) +
                                   list_brackets[len(list_brackets) // 2:])
        elif isinstance(val, dict):
            string += f'{key}=' + ':'.join([k + ':' + _list_join(':', v) for k, v in val.items()])
        else:
            # hope that key_val_dict_to_string encodes value properly
            string += key_val_dict_to_str({key: val})

        string += ' '

    # trim off final space
    return string[:-1]
