from wfl.autoparallelize import autoparallelize, autoparallelize_docstring

def _map_autopara_wrappable(atoms, map_func, args=[], kwargs={}):
    """apply an arbitrary function to a set of atomic configurations

    Parameters
    ----------
    map_func: function(Atoms, *args, **kwargs)
        function to apply
    args: list
        positional arguments to function
    kwargs: dict
        keyword arguments to function
    """
    outputs = []
    for at in atoms:
        outputs.append(map_func(at, *args, **kwargs))

    return outputs

def map(*args, **kwargs):
    return autoparallelize(_map_autopara_wrappable, *args, **kwargs)
autoparallelize_docstring(map, _map_autopara_wrappable, "Atoms")
