"""
CASTEP calculator with functionality to use both MPI and multiprocessing

"""

import os
import pathlib
import tempfile
import warnings

from ase import Atoms
from ase.calculators.calculator import CalculationFailed
from ase.calculators.castep import Castep

from .utils import clean_rundir, handle_nonperiodic, save_results, clean_failed_results
from ..utils.misc import atoms_to_list

# NOMAD compatible, see https://nomad-lab.eu/prod/rae/gui/uploads
__default_keep_files = ["*.castep", "*.param", "*.cell"]
__default_properties = ["energy", "forces", "stress"]


def evaluate_op(
    atoms,
    base_rundir=None,
    dir_prefix="run_CASTEP_",
    calculator_command=None,
    calculator_kwargs=None,
    output_prefix="CASTEP_",
    properties=None,
    keep_files="default",
):
    """Evaluate a configuration with CASTEP

    Parameters
    ----------
    atoms: Atoms / list(Atoms)
        input atomic configs
    base_rundir: path-like, default os.getcwd()
        directory to put calculation directories into
    dir_prefix: str, default 'CASTEP\_run\_'
        directory name prefix for calculations
    calculator_command
    calculator_kwargs : dict
    output_prefix : str / None, default 'CASTEP\_'
        prefix for info/arrays keys, None for SinglePointCalculator
    properties : list(str), default None
        ase-compatible property names, None for default list (energy, forces, stress)
    keep_files: bool / None / "default" / list(str), default "default"
        what kind of files to keep from the run

            - True : everything kept
            - None, False : nothing kept, unless calculation fails
            - "default"   : only ones needed for NOMAD uploads ('\*.castep', '\*.param', '\*.cell')
            - list(str)   : list of file globs to save

    Returns
    -------
    atoms: Atoms or list(Atoms)
        Atoms or list(Atoms) with calculated properties
    """
    # use list of atoms in any case
    at_list = atoms_to_list(atoms)

    # default properties
    if properties is None:
        properties = __default_properties

    # keyword setup
    if calculator_kwargs is None:
        calculator_kwargs = dict()

    if base_rundir is None:
        # using the current directory
        base_rundir = os.getcwd()
    else:
        pathlib.Path(base_rundir).mkdir(parents=True, exist_ok=True)

    # set up the keywords of the calculator -- this is reused
    kwargs_this_calc = dict(
        calculator_kwargs, label=calculator_kwargs.get("label", "castep"),
    )
    if calculator_command is not None:
        kwargs_this_calc["castep_command"] = calculator_command

    if kwargs_this_calc.get("castep_pp_path", None) is not None:
        # make sure we are looking for pspot if path given
        kwargs_this_calc["find_pspots"] = True

    for at in at_list:
        # periodic calculations
        nonperiodic, properties_use = handle_nonperiodic(at, properties)
        kwargs_this_calc["calculate_stress"] = "stress" in properties_use

        # create temp dir and calculator
        rundir = tempfile.mkdtemp(dir=base_rundir, prefix=dir_prefix)
        at.calc = Castep(directory=rundir, **kwargs_this_calc)

        # calculate
        calculation_succeeded = False
        try:
            at.calc.calculate(at)
            calculation_succeeded = True
        except Exception as exc:
            # TypeError needed here until https://gitlab.com/ase/ase/-/issues/912 is resolved
            warnings.warn(f'Calculation failed with exc {exc}')

        if calculation_succeeded:
            # NOTE: this try catch should not be necessary, but ASE castep calculator does not
            # always (ever?) raise an exception when it fails.  Instead, you get things like 
            # stress being None, which lead to TypeError when save_results calls get_stress().
            try:
                save_results(at, properties_use, output_prefix)
            except TypeError:
                calculation_succeeded=False

        # clean up results, in case castep calculator returned None for some property instead of raising
        # an exception
        calculation_succeeded = clean_failed_results(at, properties_use, output_prefix, calculation_succeeded)

        if nonperiodic:
            # reset it, because Castep overwrites the PBC to True without question
            at.set_pbc(False)

        clean_rundir(rundir, keep_files, __default_keep_files, calculation_succeeded)

    if isinstance(atoms, Atoms):
        return at_list[0]
    else:
        return at_list
