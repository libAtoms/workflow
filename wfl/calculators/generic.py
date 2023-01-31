import warnings
import functools

from ase import Atoms
from ase.calculators.calculator import all_changes

from wfl.autoparallelize import autoparallelize, autoparallelize_docstring
from wfl.utils.misc import atoms_to_list
from wfl.utils.parallel import construct_calculator_picklesafe
from .utils import save_results


def _run_autopara_wrappable(atoms, calculator, properties=None, output_prefix='_auto_', verbose=False, raise_calc_exceptions=False):
    """evaluates configs using an arbitrary calculator and store results in SinglePointCalculator

    Defaults to wfl_num_inputs_per_python_subprocess=10, to avoid recreating the calculator for
    each configuration, unless calculator class defines a wfl_generic_def_autopara_info
    attribute in which case that value is used for the default.

    Parameters
    ----------
    atoms: ase.atoms.Atoms / list(Atoms)
        input configuration(s)
    calculator: Calculator / (initializer, args, kwargs)
        ASE calculator or routine to call to create calculator
    properties: list(str), default ['energy', 'forces', stress']
        Properties to request from calculator. If any are not present after calculation (e.g.
        stress for nonperiodic configurations), a warning will be printed.
    output_prefix: str, default _auto_
        String to prefix info/arrays key names where results will be stored.
        '_auto_' for automatically determining name of calculator constructor, and
        None for SinglePointCalculator instead of info/arrays.
    verbose : bool
        verbose output
    """

    if properties is None:
        properties = ['energy', 'forces', 'stress']
    calculator = construct_calculator_picklesafe(calculator)

    if output_prefix == '_auto_':
        output_prefix = calculator.__class__.__name__ + '_'

    at_out = []
    for at in atoms_to_list(atoms):
        at.calc = calculator
        calculation_succeeded = False
        try:
            # explicitly pass system_changes=all_changes because some calculators, e.g. ace.ACECalculator,
            # don't have that as default
            at.calc.calculate(at, properties=properties, system_changes=all_changes)
            calculation_succeeded = True
            if f'{output_prefix}calculation_failed' in at.info:
                del at.info[f'{output_prefix}calculation_failed']
        except Exception as exc:
            if raise_calc_exceptions:
                raise exc
            import sys
            # pytest seems to hide these warnings for some reason
            if "pytest" in sys.modules:
                print(f'WARNING: calculation failed with exception {exc}')
            warnings.warn(f'calculation failed with exception {exc}')
            at.info[f'{output_prefix}calculation_failed'] = True

        # clean up invalid properties, will be fixed in quip Potential soon?
        if hasattr(at.calc, "results") and 'virial' in at.calc.results:
            del at.calc.results['virial']

        if calculation_succeeded:
            save_results(at, properties, output_prefix)
        else:
            # avoid maintaining the reference to the calculator
            at.calc = None

        at_out.append(at)

    if isinstance(atoms, Atoms):
        return at_out[0]
    else:
        return at_out


def run(*args, **kwargs):
    calculator = kwargs.get("calculator")
    if calculator is None:
        calculator = args[2]

    def_autopara_info = getattr(calculator, "wfl_generic_def_autopara_info", {"num_inputs_per_python_subprocess": 10})

    return autoparallelize(_run_autopara_wrappable, *args, def_autopara_info=def_autopara_info, **kwargs)
autoparallelize_docstring(run, _run_autopara_wrappable, "Atoms")
