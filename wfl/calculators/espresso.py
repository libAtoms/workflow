"""
Quantum Espresso interface
"""

import os
import shlex

from copy import deepcopy
import numpy as np

from ase.calculators.calculator import all_changes
from ase.calculators.espresso import Espresso as ASE_Espresso
from ase.io.espresso import kspacing_to_grid

from .wfl_fileio_calculator import WFLFileIOCalculator
from wfl.utils.save_calc_results import save_calc_results

# NOMAD compatible, see https://nomad-lab.eu/prod/rae/gui/uploads
_default_keep_files = ["*.pwo"]
_default_properties = ["energy", "forces", "stress"]           # done as "implemented_propertie"


class Espresso(WFLFileIOCalculator, ASE_Espresso):
    """Extension of ASE's Espresso calculator that can be used by wfl.calculators.generic

    "directory" argument cannot be present. Use rundir_prefix and workdir instead.

    Parameters
    ----------
    keep_files: bool / None / "default" / list(str), default "default"
        what kind of files to keep from the run
            - True : everything kept
            - None, False : nothing kept, unless calculation fails
            - "default"   : only ones needed for NOMAD uploads ('\*.pwo')
            - list(str)   : list of file globs to save
    rundir_prefix: str / Path, default 'run\_QE\_'
        Run directory name prefix
    workdir: str / Path, default . at calculate time
        Path in which rundir will be created.
    scratchdir: str / Path, default None
        temporary directory to execute calculations in and delete or copy back results (set by
        "keep_files") if needed.  For example, directory on a local disk with fast file I/O.

    **kwargs: arguments for ase.calculators.espresso.Espresso
    """
    implemented_properties = ["energy", "forces", "stress"]

    # new default value of num_inputs_per_python_subprocess for calculators.generic,
    # to override that function's built-in default of 10
    wfl_generic_default_autopara_info = {"num_inputs_per_python_subprocess": 1}

    def __init__(self, keep_files="default", rundir_prefix="run_QE_",
                 workdir=None, scratchdir=None, **kwargs):

        kwargs_command = deepcopy(kwargs)

        # WFLFileIOCalculator is a mixin, will call remaining superclass constructors for us
        super().__init__(keep_files=keep_files, rundir_prefix=rundir_prefix,
                         workdir=workdir, scratchdir=scratchdir, **kwargs_command)

        # we modify the parameters in self.calculate() based on the individual atoms object,
        # so let's make a copy of the initial parameters
        self.initial_parameters = deepcopy(self.parameters)


    def calculate(self, atoms=None, properties=_default_properties, system_changes=all_changes):
        """Do the calculation. Handles the working directories in addition to regular
        ASE calculation operations (writing input, executing, reading_results)
        Reimplements & extends GenericFileIOCalculator.calculate() for the development version of ASE
        or FileIOCalculator.calculate() for the v3.22.1"""

        if atoms is not None:
            self.atoms = atoms.copy()

        # this may modify self.parameters, will reset them back to initial after calculation
        self.setup_calc_params(properties)

        # from WFLFileIOCalculator
        self.setup_rundir()

        try:
            super().calculate(atoms=atoms, properties=properties, system_changes=system_changes)
            calculation_succeeded = True
            if 'DFT_FAILED_ESPRESSO' in atoms.info:
                del atoms.info['DFT_FAILED_ESPRESSO']
            if "__calculator_output_prefix" in atoms.info:
                save_calc_results(atoms, prefix=atoms.info["__calculator_output_prefix"], properties=properties)
                atoms.info["__calculator_results_saved"] = True
        except Exception as exc:
            atoms.info['DFT_FAILED_ESPRESSO'] = True
            calculation_succeeded = False
            raise exc
        finally:
            # from WFLFileIOCalculator
            self.clean_rundir(_default_keep_files, calculation_succeeded)

            # Return the parameters to what they were when the calculator was initialised.
            # There is likely a more ASE-appropriate way with self.set() and self.reset(), etc.
            self.parameters = deepcopy(self.initial_parameters)

    def setup_calc_params(self, properties):
        """Setup calculator params based on atoms structure (pbc) and requested properties
        """
        # update the parameters with the cool wfl logic
        self.parameters["tprnfor"] = "forces" in properties
        self.parameters["tstress"] = "stress" in properties

        if np.all(~self.atoms.pbc):
            # FFF -> gamma point only
            self.parameters["kpts"] = None
            self.parameters["kspacing"] = None
            self.parameters["koffset"] = False
        elif np.any(~self.atoms.pbc):
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
