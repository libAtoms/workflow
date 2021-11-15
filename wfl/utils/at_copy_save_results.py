from wfl.calculators.utils import save_results


def at_copy_save_results(at, properties=None, results_prefix=None):
    at_copy = at.copy()
    at_copy.calc = at.calc
    save_results(at_copy, properties, results_prefix=results_prefix)

    return at_copy
