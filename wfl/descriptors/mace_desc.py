import numpy as np

from wfl.autoparallelize import autoparallelize, autoparallelize_docstring
from wfl.utils.parallel import construct_calculator_picklesafe
from ase.atoms import Atoms


def _calc_autopara_wrappable(atoms, calculator, key, invariants_only=True, per_atom=False, normalize=True, force=False):
    """
    MACE descriptor 

    Parameters
    ----------
    atoms : ase.atoms.Atoms / list(Atoms)
        input configurations
    calculator : calculator
        In this case MACE calculator
    key: str
        key in Atoms.info (``not per_atom``) or Atoms.arrays (``per_atom``) to store information
    invariants_only : bool, default True
        get only invariant part of the descriptor
    per_atom: bool, default False
        calculate a local (per-atom) descriptor, as opposed to global (per-config)
    normalize: bool, default True
        normalize final vector (e.g. if contributions from multiple descriptors were concatenated)
    force: bool, default False
        overwrite key if already exists

    Returns
    -------
    atoms : Atoms or list(Atoms)
        Input configurations with descriptors in info/arrays
    """
    
    calculator = construct_calculator_picklesafe(calculator)

    if isinstance(atoms, Atoms):
        at_list = [atoms]
    else:
        at_list = atoms
        
    descriptor_encoded_atoms = []
    for i, at in enumerate(at_list):
        if key in at.info:
            if force:
                del at.info[key]
            else:
                raise RuntimeError(f'Got info key {key} which already exists, pass force=True to overwrite')
        
        if per_atom:
            at.info[key] = calculator.get_descriptors(at, invariants_only=invariants_only)
        else:
            descriptor = np.average(calculator.get_descriptors(at, invariants_only=invariants_only), axis=0)
            if normalize:
                at.info[key] = descriptor / np.linalg.norm(descriptor)
            else:
                at.info[key] = descriptor

        descriptor_encoded_atoms.append(at)
    
    return descriptor_encoded_atoms

def calculate(*args, **kwargs):
    return autoparallelize(_calc_autopara_wrappable, *args, **kwargs)
autoparallelize_docstring(calculate, _calc_autopara_wrappable, "Atoms")
