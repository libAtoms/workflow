from copy import deepcopy

class AutoparaInfo:
    """Object containing information required to autoparallelize a function

    Parameters
    ----------

    num_inputs_per_python_subprocess: int, default 1
        number of inputs passed to each call of the low level operation.

    iterable_arg: int / str, default 0
        index (int positional, str keyword) of input iterable argument in low level function

    skip_failed: bool, default True
        skip output for failed low level function calls

    initializer: (func, func_kwargs), default (None, [])
        initializer to be called when each python subprocess is started

    hash_ignore: list(str), default []
        list of arguments to ignore when doing hash of remote function arguments to determine if it's already been done

    num_python_subprocesses: int, default None
        number of python subprocesses

    remote_info: RemoteInfo, default None
        information for running remotely

    remote_label: str, default None
        string label to match to keys in remote_info dict
    """

    _kwargs = {"num_inputs_per_python_subprocess": None,
               "iterable_arg": 0,
               "skip_failed": True,
               "initializer": (None, []),
               "hash_ignore": [],
               "num_python_subprocesses": None,
               "remote_info": None,
               "remote_label": None}


    def __init__(self, **kwargs):
        # receive all args as kwargs so that we can detect if they were passed in explicitly
        for k in kwargs:
            if k not in AutoparaInfo._kwargs:
                raise ValueError(f"Invalid keyword argument in {k}")
            setattr(self, k, kwargs[k])


    def update_defaults(self, def_kwargs):
        # copy defaults dict so it can be modified (pop)
        def_kwargs = def_kwargs.copy()

        # set missing values from def_kwargs, falling back to class-predefined defaults
        for k in AutoparaInfo._kwargs:
            if not hasattr(self, k):
                setattr(self, k, def_kwargs.pop(k, AutoparaInfo._kwargs[k]))
            elif k in AutoparaInfo._kwargs:
                # it is a valid default, but was already set in AutoparaInfo, so not 
                # overwriting with defaults
                del def_kwargs[k] 


        if len(def_kwargs) != 0:
            raise ValueError(f"def_kwargs contained unknown keywords {list(def_kwargs.keys())}")


    def __str__(self):
        return (" ".join([f"{k} {getattr(self, k)}" for k in AutoparaInfo._kwargs.keys()]))
