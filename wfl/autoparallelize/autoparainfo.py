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

    _kwargs = {"num_inputs_per_python_subprocess": 1,
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
        """Starting from object passed by user at runtime, update all unspecified fields to the defaults
        specified when wrapping function, otherwise to class-wide defaults
        """
        # copy defaults dict so it can be modified (pop)
        def_kwargs = def_kwargs.copy()

        # set missing values from def_kwargs, falling back to class-predefined defaults
        # remove each as its used from def_kwargs, so any remaining can be detected as invalid
        for k in AutoparaInfo._kwargs:
            if not hasattr(self, k):
                # user hasn't set this attribute, set it from wrapper or class-wide default
                setattr(self, k, def_kwargs.pop(k, AutoparaInfo._kwargs[k]))
            else:
                # user has set this, still remove from def_kwargs to facilitate check for invalid
                # def_kwargs keys below
                if k in def_kwargs:
                    del def_kwargs[k]

        if len(def_kwargs) != 0:
            raise ValueError(f"def_kwargs contained unknown keywords {list(def_kwargs.keys())}")


    def __str__(self):
        return (" ".join([f"{k} {getattr(self, k)}" for k in AutoparaInfo._kwargs.keys()]))
