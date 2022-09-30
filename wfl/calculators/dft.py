"""
This is a unified wrapper for Plane Wave DFT codes.

Currently implemented codes:
- CASTEP
- VASP
- Quantum Espresso
"""
from wfl.autoparallelize import autoparallelize, autoparallelize_docstring
from wfl.calculators import castep, vasp 

def evaluate_dft(*args, **kwargs):
    """evaluate configurations with a DFT calculator

    Parameters
    ----------
    *args:
        positional arguments for actual DFT calculator
    **kwargs:
        keyword args for actual DFT calculator
        must include calculator_name in ["CASTEP", "VASP"]

    Returns
    -------
    ConfigSet of configurations with calculated properties
    """

    try:
        calculator_name = kwargs.pop("calculator_name")
    except KeyError as exc:
        raise ValueError("evaluate_dft requires 'calculator_name' in kwargs") from exc

    # defaults
    if kwargs.get("dir_prefix") is None:
        kwargs["dir_prefix"] = f"run_{calculator_name}_"

    # choose the calculator
    if calculator_name == "CASTEP":
        op = castep.evaluate_autopara_wrappable
    elif calculator_name == "VASP":
        raise ValueError(f"wfl.calculators.vasp.Vasp is compatible with wfl.calculators.generic.run()")
    elif calculator_name == "QE":
        raise ValueError(f"wfl.calculators.espresso.Espresso is compatible with wfl.calculators.generic.run()")
    else:
        raise ValueError(f"Calculator name `{calculator_name}` not understood")

    # run the calculation in parallel
    return autoparallelize(op, *args, def_autopara_info={"num_inputs_per_python_subprocess": 1}, **kwargs)

# NOTE: arguments depend on actual DFT calculator, so no way to fix the docstring
# evaluate_dft.__doc__ = autoparallelize_docstring(????, "Atoms")
