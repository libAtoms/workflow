"""
FHI-aims calculator with functionality to use both MPI and multiprocessing

"""
import time

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


# used for collecting time and numbe of scf cycles by default
def reverse_search_for(lines_obj, keys, line_start=0):
    for ll, line in enumerate(lines_obj[line_start:][::-1]):
        if any([key in line for key in keys]):
            return len(lines_obj) - ll - 1


def evaluate_autopara_wrappable(
    atoms,
    workdir_root=None,
    dir_prefix="run_AIMS_",
    calculator_command=None,
    calculator_kwargs=None,
    output_prefix="AIMS_",
    properties=None,
    keep_files="default",
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
    
    note: 
        Aims properly supports both periodic calcuations and non-periodic. 
        this calculator will fail if p[arameters are inconsisten - eg. if kpoints are specified and pbc=False
        no support for semi-periodic yet.
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
        
    if workdir_root is None:
        workdir_root = os.getcwd()
    else:
        pathlib.Path(workdir_root).mkdir(parents=True, exist_ok=True)

    # set up the keywords of the calculator -- this is reused
    calculator_kwargs = dict(calculator_kwargs)
    
    if not "species_dir" in calculator_kwargs:
        raise ValueError('no species directory set')

    """ Required contents of calculator_kwargs: 
        all calculations:
            'xc'
            'aims_command'
            'species_dir'     

        periodic calculations:
            'k_grid' || 'kpts'

        computing forces / stress:
            'compute_forces' / 'compute_analytical_stress'
    """

    # check for any missmatch between properties and kwargs
    pbc = at_list[0].pbc
    assert(pbc[0] == pbc[1] and pbc[0] == pbc[2])
    pbc = pbc[0]

    assert(all([at.pbc[0] == pbc for at in at_list]))

    has_kdensity = "k_grid_density" in calculator_kwargs
    requires_stresses = "compute_analytical_stress" in calculator_kwargs
    assert(has_kdensity == pbc and requires_stresses == pbc)

    # don't do this
    if 'aims_command' in calculator_kwargs:
        raise ValueError('unexpected key aims_command in the aims kwargs dicitionary')

    for at in at_list:
        # create temp dir and calculator
        rundir = tempfile.mkdtemp(dir=workdir_root, prefix=dir_prefix)
        prof = AimsProfile(calculator_command.split())
        at.calc = Aims(profile=prof, directory=rundir, **calculator_kwargs)

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

        # add time and num scf to atoms
        try:
            with open(os.path.join(rundir, 'aims.out')) as f:
                lines = f.readlines()

                line_start = reverse_search_for(lines, ["Detailed time accounting"]) + 1
                tot_time = float(lines[line_start].split(":")[-1].strip().split()[0])
                at.info['calculation_time'] = tot_time

                line_start = reverse_search_for(lines, ["| Number of self-consistency cycles"])
                num_scf = float(lines[line_start].split(":")[-1].strip().split()[0])
                at.info['number_of_scf_cylces'] = num_scf
        except:
            pass

    if isinstance(atoms, Atoms):
        return at_list[0]
    else:
        return at_list
