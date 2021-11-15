from ase.calculators.calculator import Calculator


def construct_calculator_picklesafe(calculator):
    """Constructs a calculator safe with multiprocessing.Pool

    Trick: pass a recipe only and create the calculator in the thread created, instead of trying to pickle the entire
    object when creating the pool.

    Taken from minim.py:run_op

    Parameters
    ----------
    calculator: Calculator / (initializer, args, kwargs)
        ASE calculator or routine to call to create calculator

    Returns
    -------
    calculator: Calculator
        ase calculator object

    """

    if isinstance(calculator, Calculator):
        return calculator
    else:
        if len(calculator) != 3:
            raise RuntimeError('calculator \'{}\' must be (calc_constructor, args, kwargs)'.format(calculator))

        if not callable(calculator[0]):
            raise RuntimeError(
                'calculator \'{}\' : first element is not callable, cannot construct a calculator'.format(calculator))

        if calculator[1] is None:
            c_args = []
        else:
            c_args = calculator[1]
        if calculator[2] is None:
            c_kwargs = {}
        else:
            c_kwargs = calculator[2]

        return calculator[0](*c_args, **c_kwargs)
