import warnings
import numpy as np
from ase.calculators.calculator import all_properties, PropertyNotImplementedError
from ase.calculators.singlepoint import SinglePointCalculator

from wfl.utils.file_utils import clean_dir

per_atom_properties = ['forces', 'stresses', 'charges', 'magmoms', 'energies']
per_config_properties = ['energy', 'stress', 'dipole', 'magmom', 'free_energy']


def handle_nonperiodic(atoms, properties, allow_mixed=False):
    """prepare for a calculation by filtering out stress if nonperiodic

    Parameters
    ----------
    atoms: Atoms
        input configuration
    properties: list(str)
        list of properties to calculate
    allow_mixed: bool, default=False
        allow mixed periodicity, ie. not only TTT and FFF

    Returns
    -------
    nonperiodic: bool
    use_properties: list
        list of properties, filtering out stress for nonperiodic systems
    """

    use_properties = list(properties)
    nonperiodic = False

    if np.all(atoms.pbc):
        # keep stress
        pass
    else:
        nonperiodic = True
        if 'stress' in use_properties:
            use_properties.remove('stress')
        if 'stresses' in use_properties:
            use_properties.remove('stresses')

        if np.any(atoms.pbc) and not allow_mixed:
            raise RuntimeError(f'atoms.pbc {atoms.pbc} neither all T or all F')

    return nonperiodic, use_properties


def save_results(atoms, properties, results_prefix=None):
    """saves results of a calculation in a SinglePointCalculator or info/arrays keys

    Parameters
    ----------
    atoms: Atoms
        configuration
    properties: list(str) or None
        list of calculated properties to save, or None to use atoms.calc.results dict keys
    results_prefix: str / None , default None
        if None, store in SinglePointCalculator, else store in results_prefix+<property>.
        str with length 0 is forbidden
    """

    if isinstance(results_prefix, str) and len(results_prefix) == 0:
        raise ValueError('Refusing to save calculator results into info/arrays fields with no prefix,'
                         ' too much chance of confusion with ASE extxyz reading/writing and conversion'
                         ' to SinglePointCalculator')

    # this would be simpler if we could just use calc.results, but some (e.g. Castep) don't use it
    if properties is None:
        # This will not work for calculators like Castep that (as of some point at least) do not use 
        # results dict.  Such calculators will fail below, in the "if 'energy' in properties" statement.
        properties = list(atoms.calc.results.keys())

    # clean up for saving in info/arrays
    if results_prefix is not None:
        for p in per_config_properties:
            if results_prefix + p in atoms.info:
                del atoms.info[results_prefix + p]
        for p in per_atom_properties:
            if results_prefix + p in atoms.arrays:
                del atoms.arrays[results_prefix + p]

    # copy per-config results
    config_results = {}
    atoms.calc.atoms = atoms
    if 'energy' in properties:
        try:
            config_results['energy'] = atoms.get_potential_energy(force_consistent=True)
        except PropertyNotImplementedError:
            config_results['energy'] = atoms.get_potential_energy()
    if 'stress' in properties:
        # Quantum Espresso doesn't calculate stress, even if asked for, if pbc=False.
        try:
            config_results['stress'] = atoms.get_stress()
        except PropertyNotImplementedError:
            warnings.warn(f'"stress" was asked for, but not found in results.')
    if 'dipole' in properties:
        config_results['dipole'] = atoms.get_dipole_moment()
    if 'magmom' in properties:
        config_results['magmom'] = atoms.get_magnetic_moment()

    # copy per-atom results
    atoms_results = {}
    if 'forces' in properties:
        atoms_results['forces'] = atoms.get_forces()
    if 'stresses' in properties:
        atoms_results['stresses'] = atoms.get_stresses()
    if 'charges' in properties:
        atoms_results['charges'] = atoms.get_charges()
    if 'magmoms' in properties:
        atoms_results['magmoms'] = atoms.get_magnetic_moments()
    if 'energies' in properties:
        atoms_results['energies'] = atoms.get_potential_energies()

    if "extra_results" in dir(atoms.calc):
        if results_prefix is None and (len(atoms.calc.extra_results.get("config", {})) > 0 or
                                       len(atoms.calc.extra_results.get("atoms", {})) > 0):
            raise ValueError('Refusing to save calculator results into info/arrays fields with no prefix,'
                            ' too much chance of confusion with ASE extxyz reading/writing and conversion'
                            ' to SinglePointCalculator') 

        for key, vals in atoms.calc.extra_results["config"].items():
            config_results[key] = vals

        for key, vals in atoms.calc.extra_results["atoms"].items():
            atoms_results[key] = vals

        # Update atoms' positions if geometry was optimised
        if "relaxed_positions" in atoms.calc.extra_results:
            atoms.set_positions(atoms.calc.extra_results["relaxed_positions"])

    # write to Atoms
    if results_prefix is None:
        # Filter out nonstandard properties that will cause SinglePointCalculator to fail
        config_results = {k: v for k, v in config_results.items() if k in all_properties}
        atoms_results = {k: v for k, v in atoms_results.items() if k in all_properties}
        atoms.calc = SinglePointCalculator(atoms, **config_results, **atoms_results)
    else:
        atoms.calc = None
        for p, v in config_results.items():
            atoms.info[results_prefix + p] = v
        for p, v in atoms_results.items():
            atoms.new_array(results_prefix + p, v)


def clean_rundir(rundir, keep_files, default_keep_files, calculation_succeeded):
    """clean up a run directory from a file-based calculator

    Parameters
    ----------
    rundir: str
        path to run dir
    keep_files: 'default' / list(str) / '*' / bool / None
        files to keep, None or False for nothing, '*' or True for all
    default_keep_files: list(str)
        files to keep if keep_files == 'default' or calculation_succeeded is False
    calculation_succeeded: bool
    """
    if keep_files == 'default' or not calculation_succeeded:
        clean_dir(rundir, default_keep_files, force=False)
    elif not keep_files:
        clean_dir(rundir, False, force=False)
    else:
        clean_dir(rundir, keep_files, force=False)
