"""
FHI-aims calculator with functionality to use both MPI and multiprocessing

"""

import os
import pathlib
import tempfile

from ase import Atoms
from ase.calculators.calculator import all_changes, CalculationFailed
from ase.calculators.aims import Aims, AimsProfile


from .utils import clean_rundir, handle_nonperiodic, save_results, clean_failed_results
from ..utils.misc import atoms_to_list

# NOMAD compatible, see https://nomad-lab.eu/prod/rae/gui/uploads
__default_keep_files = ["*"]
__default_properties = ["energy", "forces", "stress"]


def evaluate_op(
    atoms,
    base_rundir=None,
    dir_prefix="AIMS_run_",
    calculator_command=None,
    calculator_kwargs=None,
    output_prefix="aims_",
    properties=None,
    keep_files="default"
):
    """Evaluate a configuration with FHI-aims

    Parameters
    ----------
    atoms: Atoms / list(Atoms)
        input atomic configs
    base_rundir: path-like, default os.getcwd()
        directory to put calculation directories into
    dir_prefix: str, default 'AIMS_run_'
        directory name prefix for calculations
    calculator_command
    calculator_kwargs : dict
    output_prefix : str / None, default 'aims_'
        prefix for info/arrays keys, None for SinglePointCalculator
    properties : list(str), default None
        ase-compatible property names, None for default list (energy, forces, stress)
    keep_files: bool / None / "default" / list(str), default "default"
        what kind of files to keep from the run
            True : everything kept
            None, False : nothing kept, unless calculation fails
            "default"   : only ones needed for NOMAD uploads ('*.castep', '*.param', '*.cell')
            list(str)   : list of file globs to save

    Returns
    -------
        Atoms or list(Atoms) with calculated properties
    """
    # use list of atoms in any case
    at_list = atoms_to_list(atoms)

    # default properties
    if properties is None:
        properties = __default_properties

    # keyword setup
    if calculator_kwargs is None:
        raise ValueError('did not find aims kwargs')

    if calculator_command is None:
        raise ValueError('calculator command not specified')
        
    if base_rundir is None:
        base_rundir = os.getcwd()
    else:
        pathlib.Path(base_rundir).mkdir(parents=True, exist_ok=True)

    # set up the keywords of the calculator -- this is reused
    aims_calc_kwargs = dict(calculator_kwargs)
    
    #ncores_per_task = os.environ["EXPYRE_NCORES_PER_TASK"]    
    #if calculator_command is not None:
    #    mpi_calculator_command = "mpirun -n " + str(ncores_per_task) + " " + calculator_command
    #    kwargs_this_calc["command"] = mpi_calculator_command

    #if calculator_command is not None:
    #    aims_calc_kwargs["command"] = calculator_command
    
    if not "species_dir" in aims_calc_kwargs:
        raise ValueError('no species directory set')

    """ Required contents of aims_calc_kwargs: 
        all calculations:
            'xc'
            'aims_command'
            'species_dir'     

        periodic calculations:
            'k_grid' || 'kpts'

        computing forces / stress:
            'compute_forces' / 'compute_analytical_stress'

    note: the following assumes that the same calculation is not perfomed over 
    a config set containing both periodic and non-perodic systems
    """

    # check for any missmatch between properties and kwargs
    if (('compute_forces' in aims_calc_kwargs) and ('forces' not in properties) or 
        ('compute_forces' not in aims_calc_kwargs) and ('forces' in properties)):
        raise ValueError('compute_forces and contents of properties are inconsistent')

    if (('compute_analytical_stress' in aims_calc_kwargs) and ('stress' not in properties) or 
        ('compute_analytical_stress' not in aims_calc_kwargs) and ('stress' in properties)):
        raise ValueError('compute_analytical_stress and contents of properties are inconsistent')

    # can specifiy
    print(calculator_command)
    if 'aims_command' in aims_calc_kwargs:
        raise ValueError('unexpected key aims_command in the aims kwargs dicitionary')

    for at in at_list:
        # create temp dir and calculator
        rundir = tempfile.mkdtemp(dir=base_rundir, prefix=dir_prefix)
        prof = AimsProfile(calculator_command.split())
        at.calc = Aims(profile=prof, directory=rundir, **aims_calc_kwargs)

        # calculate
        calculation_succeeded = False
        try:
            at.calc.calculate(at, properties=properties, system_changes=all_changes)
            calculation_succeeded = True
        except (CalculationFailed, TypeError):
            pass

        if calculation_succeeded:
            # NOTE: this try catch should not be necessary, but ASE castep calculator does not
            # always (ever?) raise an exception when it fails.  Instead, you get things like 
            # stress being None, which lead to TypeError when save_results calls get_stress().
            try:
                save_results(at, properties, output_prefix)
            except TypeError:
                calculation_succeeded=False

        # clean up results, in case castep calculator returned None for some property instead of raising
        # an exception
        calculation_succeeded = clean_failed_results(at, properties, output_prefix, calculation_succeeded)

        clean_rundir(rundir, keep_files, __default_keep_files, calculation_succeeded)

    if isinstance(atoms, Atoms):
        return at_list[0]
    else:
        return at_list
