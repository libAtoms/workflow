"""
Quantum Espresso interface
"""

import os
import tempfile
import subprocess
import warnings
import shutil

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
default_keep_files = ["*.pwo"]
default_properties = ["energy", "forces", "stress"]           # done as "implemented_propertie"


class Espresso(ase.calculators.espresso.Espresso):
    """Extension of ASE's Espresso calculator

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

    **kwargs: arguments for ase.calculators.espresso.Espresso
    """
    implemented_properties = ["energy", "forces", "stress"]

    default_parameters = {} 

    # default value of wfl_num_inputs_per_python_subprocess for calculators.generic,
    # to override that function's built-in default of 10
    wfl_generic_num_inputs_per_python_subprocess = 1

    def __init__(self, atoms=None, keep_files="default", 
                 dir_prefix="run_QE_", scratch_path=None,
                 calculator_command=None, **kwargs):
            
        super(Espresso, self).__init__(**kwargs)

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


    def calculate(self, atoms=None, properties=default_properties , system_changes=all_changes):
        """Does the calculation. Handles the working directories in addition to regular 
        ASE calculation operations (writing input, executing, reading_results) 
        Reimplements & extends GenericFileIOCalculator.calculate() for the development version of ASE
        or FileIOCalculator.calculate() for the v3.22.1"""

        if atoms is not None:
            self.atoms = atoms.copy()

        properties = self.setup_params_for_this_calc(properties)

        if self.scratch_path is not None:
            directory = tempfile.mkdtemp(dir=self.scratch_path, prefix=self.dir_prefix)
        else:
            directory = tempfile.mkdtemp(dir=self.workdir_root, prefix=self.dir_prefix)
        self.directory = Path(directory)

        try:
            super().calculate(atoms=atoms, properties=properties,system_changes=system_changes) 
            calculation_succeeded=True
        except Exception as e:
            calculation_succeeded=False
            raise e
        finally:
            # when exception is raised, `calculation_succeeded` is set to False, 
            # the following code is executed and exception is re-raised. 
            clean_rundir(directory, self.keep_files, default_keep_files, calculation_succeeded)
            if self.scratch_path is not None and Path(directory).exists():
                shutil.move(directory, self.workdir_root)

            # Return the parameters to what they were when the calculator was initialised.
            # There is likely a more ASE-appropriate way with self.set() and self.reset(), etc. 
            self.parameters = deepcopy(self.initial_parameters)

    def setup_params_for_this_calc(self, properties):

        # first determine if we do a non-periodic calculation. 
        # and update the properties that we will use. 
        nonperiodic, properties = handle_nonperiodic(self.atoms, properties, allow_mixed=True)

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

