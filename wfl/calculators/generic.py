import warnings

from ase import Atoms
from ase.calculators.calculator import all_changes

from wfl.pipeline import iterable_loop
from wfl.utils.misc import atoms_to_list
from wfl.utils.parallel import construct_calculator_picklesafe
from .utils import save_results


# run that operates on ConfigSet, for multiprocessing
def run(inputs, outputs, calculator, properties=None, output_prefix='_auto_', chunksize=10, verbose=False):
    if properties is None:
        properties = ['energy', 'forces', 'stress']
    return iterable_loop(iterable=inputs, configset_out=outputs, op=run_op, chunksize=chunksize,
                         calculator=calculator, properties=properties, output_prefix=output_prefix,
                         verbose=verbose)


def run_op(atoms, calculator, properties=None, output_prefix='_auto_', verbose=False):
    """evaluates configs using an arbitrary calculator and store results in SinglePointCalculator

    Parameters
    ----------
    atoms: list(Atoms)
        input configs
    calculator: Calculator / (initializer, args, kwargs)
        ASE calculator or routine to call to create calculator
    properties: list(str), default ['energy']
        properties to request from calculator
    output_prefix: str, default _auto_
        string to prefix info/arrays key names where results will be stored.
        '_auto_' for automatically determining name of calculator constructor, and
        None for SinglePointCalculator instead of info/arrays
    verbose : bool

    Returns
    -------
    list(Atoms) 
        Evaluated configurations
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
            warnings.warn(f'calculation failed with exception {exc}')
            at.info[f'{output_prefix}calculation_failed'] = True

        # clean up invalid properties, will be fixed in quip Potential soon?
        if 'virial' in at.clc.results:
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
