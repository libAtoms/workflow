"""
Quantum Espresso interface
"""

import os
import tempfile
import subprocess
import warnings

from copy import deepcopy
from pathlib import Path
import numpy as np

from ase import Atoms
from ase.calculators.calculator import all_changes, CalculationFailed, Calculator
import ase.calculators.espresso
try:
    from ase.calculators.espresso import EspressoProfile
except ImportError:
    EspressoProfile = None
from ase.io.espresso import kspacing_to_grid

from .utils import clean_rundir, handle_nonperiodic, save_results
from ..utils.misc import atoms_to_list

# NOMAD compatible, see https://nomad-lab.eu/prod/rae/gui/uploads
__default_keep_files = ["*.pwo"]
__default_properties = ["energy", "forces", "stress"]           # done as "implemented_propertie"


class Espresso(ase.calculators.espresso.Espresso):
    """Extension of ASE's Espresso calculator

    workdir_root: path-like, default os.getcwd()
        directory to put calculation directories into
    dir_prefix: str, default 'QE-run\_'
        directory name prefix for calculations
    keep_files: bool / None / "default" / list(str), default "default"
        what kind of files to keep from the run
            - True : everything kept
            - None, False : nothing kept, unless calculation fails
            - "default"   : only ones needed for NOMAD uploads ('\*.pwo')
            - list(str)   : list of file globs to save
    scratch_path: str, default None
        temporary directory to execute calculations in and delete
        or copy back results if needed (set by "keep_files"). 
        For example, directory on an SSD with fast file io. 
    calculator_command: str
        command for QE, without any prefix or redirection set.
        for example: "mpirun -n 4 /path/to/pw.x"

        4. see what is done in the dft calculator 

    """
    implemented_properties = ["energy", "forces", "stress"]

    default_parameters = {} 

    def __init__(self, atoms=None, keep_files="default", 
                 dir_prefix="run_QE_", scratch_path=None,
                 calculator_command=None, **kwargs):
                 # calculator command?
            
        # parameters are set from kwargs at the Calculator level
        super(Espresso, self).__init__()

        self.keep_files = keep_files
        self.scratch_path = scratch_path
        # self.directory is overwritten in self.calculate, so let's just keep track
        self.dir_prefix = dir_prefix
        self.workdir_root = Path(self.directory) / (self.dir_prefix + 'FileIOCalc_files') 
        self.workdir_root.mkdir(parents=True, exist_ok=True)

        if calculator_command is not None: 
            if EspressoProfile is None:
                # older syntax
                self.parameters["command"] = f"{calculator_command} -in PREFIX.pwi > PREFIX.pwo"
            else:
                # newer syntax
                self.parameters['profile'] = EspressoProfile(argv=calculator_command.split())


        # we modify the parameters in self.calculate() based on the individual atoms object, 
        # so let's make a copy of the initial parameters
        self.initial_parameters = deepcopy(self.parameters)

    def calculate(self, atoms=None, properties=["energy", "forces", "stress"] , system_cahges=all_changes):
        """Does the calculation. Handles the working directories in addition to regular 
        ASE calculation operations (writing input, executing, reading_results) """

        # just makes the directories, essentially and sets self.atoms with given atoms, if not none. 
        # Properties and system changes are ignored. So let's do it with default properties, even 
        # though they are updated later for just this calculation. 
        Calculator.calculate(self, atoms, properties, system_changes)

        properties = self.setup_params_for_this_calc(properties)


        if self.scratch_path is not None:
            self.directory = tempfile.mkdtemp(dir=self.scratch_path, prefix=self.dir_prefix)
        else:
            self.directory = tempfile.mkdtemp(dir=self.workdir_root, prefix=self.dir_prefix)


        try:
            self.write_input(self.atoms, properties, system_cahges)
            self.execute()
            self.read_results()
            calculation_succeeded=True
        except Exception as e:
            calculation_succeeded=False
            raise e
        finally:
            # when exception is raised, `calculation_succeeded` is set to False, 
            # the following code is executed and exception is re-raised. 
            clean_rundir(self.directory, self.keep_files, default_keep_files, calculation_succeeded)
            if self.scratch_path is not None and Path(self.directory).exists():
                shutil.move(self.directory, self.workdir_root)

            # return the parameters to what they were when the calculator was initialised
            # there is likely a more ASE-appropriate way with self.set() and self.reset(), etc. 
            self.parameters = deepcopy(self.initial_parameters)


    def setup_params_for_this_calc(self, properties):

        # first determine if we do a non-periodic calculation. 
        # and update the properties that we will use. 
        nonperiodic, properties = handle_nonperiodic(atoms, properties, allow_mixed=True)

        # update the parameters with the cool wfl logic 
        self.parameters["tprnfor"] = "forces" in properties
        self.parameters["tstress"] = "stress" in properties

        if nonperiodic:
            if not np.any(self.atoms.get_pbc()):
                # FFF -> gamma point only
                self.parameters["kpts"] = None
                self.parameters["kspacing"] = None
                self.parameters["koffset"] = False
            else:
                # mixed T & F
                if "kspacing" in self.parameters:
                    # need to create the grid,
                    # `kspacing` overwrites `kpts`
                    # we set it in there and
                    self.parameters["kpts"] = kspacing_to_grid(
                        self.atoms, spacing=self.parameters["kspacing"] / (2.0 * np.pi)
                    )

                # kspacing None anyways
                self.parameters["kspacing"] = None

                # original, or overwritten from kspacing
                if "kpts" in self.parameters:
                    # any no-periodic direction has 1 k-point only
                    kpts = np.array(self.parameters["kpts"])
                    kpts[~self.atoms.get_pbc()] = 1
                    self.parameters["kpts"] = tuple(kpts)

                # k-point offset
                k_offset = self.parameters.get("koffset", False)
                if k_offset is True:
                    k_offset = (1, 1, 1)

                if k_offset:
                    # set to zero on any non-periodic ones
                    k_offset = np.array(k_offset)
                    k_offset[~self.atoms.get_pbc()] = 0
                    self.parameters["koffset"] = tuple(k_offset)
                else:
                    self.parameters["koffset"] = k_offset        

        return properties 



    def cleanup(self):
        """Clean all (empty) directories that could not have been removed
        immediately after the calculation, for example, because other parallel
        process might be using them.
        Done because `self.workdir_root` gets created upon initialisation, but we
        can't ever be sure it's not needed anymore, so let's not do it automatically."""
        if any(self.workdir_root.iterdir()):
            print(f'{self.workdir_root.name} is not empty, not removing')
        else:
            self.workdir_root.rmdir()


def evaluate_autopara_wrappable(
    atoms,
    workdir_root=None,          # done later 
    dir_prefix="run_QE_",       # done above
    calculator_command=None,    # done above 
    calculator_kwargs=None,     # could set some defaults?
    output_prefix="QE_",        # done in generic  
    properties=None,            # done in generic
    keep_files="default",       # done above
):
    """Evaluate a configuration with Quantum Espresso

    Parameters
    ----------
    atoms: Atoms / list(Atoms)
        input atomic configs
    workdir_root: path-like, default os.getcwd()
        directory to put calculation directories into
    dir_prefix: str, default 'QE-run\_'
        directory name prefix for calculations
    calculator_command: str
        command for QE, without any prefix or redirection set.
        for example: "mpirun -n 4 /path/to/pw.x"
    calculator_kwargs : dict
    output_prefix : str / None, default 'QE\_'
        prefix for info/arrays keys, None for SinglePointCalculator
    properties : list(str), default None
        ase-compatible property names, None for default list (energy, forces, stress)
    keep_files: bool / None / "default" / list(str), default "default"
        what kind of files to keep from the run
            - True : everything kept
            - None, False : nothing kept, unless calculation fails
            - "default"   : only ones needed for NOMAD uploads ('\*.pwo')
            - list(str)   : list of file globs to save

    Returns
    -------
        Atoms or list(Atoms) with calculated properties
    """
    # use list of atoms in any case                                                     # done in generic
    at_list = atoms_to_list(atoms)

    # default properties
    if properties is None:                                                              # done in generic
        properties = __default_properties

    # keyword setup
    if calculator_kwargs is None:                                                       # Not sure whether/how to check for this? 
        raise ValueError("QE will not perform a calculation without settings given!")

    if workdir_root is None:
        # using the current directory
        workdir_root = os.getcwd()
    else:
        Path(workdir_root).mkdir(parents=True, exist_ok=True)

    for at in at_list:

        # periodicity
        kwargs_this_calc, properties_use = qe_kpoints_and_kwargs(                           # done in calculate()
            at, calculator_kwargs, properties
        )

        if calculator_command is not None:                                                  # done in __init__
            if EspressoProfile is None:
                # older syntax
                kwargs_this_calc["command"] = f"{calculator_command} -in PREFIX.pwi > PREFIX.pwo"
            else:
                # newer syntax
                kwargs_this_calc['profile'] = EspressoProfile(argv=calculator_command.split())

        # create temp dir and calculator                                                    # done in calculate()
        rundir = tempfile.mkdtemp(dir=workdir_root, prefix=dir_prefix)
        at.calc = Espresso(directory=rundir, **kwargs_this_calc)

        # calculate
        calculation_succeeded = False
        try:
            at.calc.calculate(at, properties=properties_use, system_changes=all_changes)
            calculation_succeeded = True                                                    # done at calculate()
            if 'DFT_FAILED_ESPRESSO' in at.info:                                            # done at generic.py
                del at.info['DFT_FAILED_ESPRESSO'] 
        except Exception as exc:                                                            # done at generic.py
            # CalculationFailed is probably what's supposed to be returned
            # for failed convergence, Espresso currently returns subprocess.CalledProcessError
            #     since pw.x returns a non-zero status
            warnings.warn(f'Calculation failed with exc {exc}')
            at.info['DFT_FAILED_ESPRESSO'] = True

        # save results
        if calculation_succeeded:                                                           # done at generic.py
            save_results(at, properties_use, output_prefix)

        # clean run directory
        clean_rundir(rundir, keep_files, __default_keep_files, calculation_succeeded)

    if isinstance(atoms, Atoms):                                                           # done at generic.py
        return at_list[0]
    else:
        return at_list


def qe_kpoints_and_kwargs(
    atoms: Atoms, kwargs: dict, properties: list,
):
    """Handle K-Points in QE for any periodicity

    - stress is only calculated if all directions are periodic
    - gamma point is used if none of the directions are periodic
    - single K-Point is used in any direction which is not periodic

    Parameters
    ----------
    atoms: Atoms
    kwargs: dict
        QE calculator's intended keyword arguments
    properties: list
        ASE-compatible property name list for calculation

    Returns
    -------
    modified_kwargs: dict
    properties_use: list

    """
    # periodicity, allowing mixed periodicity
    nonperiodic, properties_use = handle_nonperiodic(                               # done as first thing at calculate()
        atoms, properties, allow_mixed=True
    )

    # a copy of the parameters that we are modifying for each calculation
    # EG: not a copy??
    modified_kwargs = dict(kwargs)                                                  # modify in-place and reset after calculation is done.  

    # stress and force calculations need keys
    modified_kwargs["tprnfor"] = "forces" in properties_use
    modified_kwargs["tstress"] = "stress" in properties_use

    if nonperiodic:
        if not np.any(atoms.get_pbc()):
            # FFF -> gamma point only
            modified_kwargs["kpts"] = None
            modified_kwargs["kspacing"] = None
            modified_kwargs["koffset"] = False
        else:
            # mixed T & F
            if "kspacing" in kwargs:
                # need to create the grid,
                # `kspacing` overwrites `kpts`
                # we set it in there and
                modified_kwargs["kpts"] = kspacing_to_grid(
                    atoms, spacing=kwargs["kspacing"] / (2.0 * np.pi)
                )

            # kspacing None anyways
            modified_kwargs["kspacing"] = None

            # original, or overwritten from kspacing
            if "kpts" in modified_kwargs:
                # any no-periodic direction has 1 k-point only
                kpts = np.array(modified_kwargs["kpts"])
                kpts[~atoms.get_pbc()] = 1
                modified_kwargs["kpts"] = tuple(kpts)

            # k-point offset
            k_offset = kwargs.get("koffset", False)
            if k_offset is True:
                k_offset = (1, 1, 1)

            if k_offset:
                # set to zero on any non-periodic ones
                k_offset = np.array(k_offset)
                k_offset[~atoms.get_pbc()] = 0
                modified_kwargs["koffset"] = tuple(k_offset)
            else:
                modified_kwargs["koffset"] = k_offset

    return modified_kwargs, properties_use
