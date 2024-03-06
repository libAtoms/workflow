"""
Quantum Espresso interface
"""

import os
import shlex

from copy import deepcopy
import numpy as np

from ase.calculators.calculator import all_changes
from ase.calculators.espresso import Espresso as ASE_Espresso
try:
    from ase.calculators.espresso import EspressoProfile
except ImportError:
    EspressoProfile = None
from ase.io.espresso import kspacing_to_grid

from .wfl_fileio_calculator import WFLFileIOCalculator
from .utils import save_results, parse_genericfileio_profile_argv

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
    calculator_exec: str
        command for QE, without any prefix or redirection set.
        for example: "mpirun -n 4 /path/to/pw.x"
        mutually exclusive with "command" with "profile"

    **kwargs: arguments for ase.calculators.espresso.Espresso
    """
    implemented_properties = ["energy", "forces", "stress"]

    # new default value of num_inputs_per_python_subprocess for calculators.generic,
    # to override that function's built-in default of 10
    wfl_generic_default_autopara_info = {"num_inputs_per_python_subprocess": 1}

    def __init__(self, keep_files="default", rundir_prefix="run_QE_",
                 workdir=None, scratchdir=None,
                 calculator_exec=None, **kwargs):

        kwargs_command = deepcopy(kwargs)

        # check for various Espresso versions
        # NOTE: should we be doing this much massaging of inputs, or should we make the user keep up
        # with their ASE Espresso version?
        if EspressoProfile is not None:
            # new version, command and ASE_ESPRESSO_COMMAND deprecated
            if "command" in kwargs_command:
                raise ValueError("Espresso calculator defines EspressoProfile, but deprecated 'command' arg was passed")

            if calculator_exec is not None:
                # check for conflicts, wrong format
                if "profile" in kwargs_command:
                    raise ValueError("Cannot specify both calculator_exec and profile")
                if " -in " in calculator_exec:
                    raise ValueError("calculator_exec should not include espresso command line arguments such as ' -in PREFIX.pwi'")

                # newer syntax, but pass binary without a keyword (which changed from "argv" to "exc"
                # to "binary" over time), assuming it's first argument
                argv = shlex.split(calculator_exec)
                try:
                    kwargs_command["profile"] = EspressoProfile(argv=argv)
                except TypeError:
                    binary, parallel_info = parse_genericfileio_profile_argv(argv)
                    # argument names keep changing (e.g. pseudo_path -> pseudo_dir), just pass first two as positional
                    # and hope order doesn't change
                    if "pseudo_dir" not in kwargs_command:
                        raise ValueError(f"calculator_exec also requires pseudo_dir to create EspressoProfile")
                    kwargs_command["profile"] = EspressoProfile(binary, kwargs_command.pop("pseudo_dir"),
                                                                parallel_info=parallel_info)
            elif "profile" not in kwargs_command:
                raise ValueError("EspressoProfile is defined but neither calculator_exec nor profile was specified")

            # better be defined by now
            assert "profile" in kwargs_command
        else:
            # old (pre EspressoProfile) version
            if "profile" in kwargs_command:
                raise ValueError("EspressoProfile is not defined (old version) but profile was passed")

            if calculator_exec is not None:
                if "command" in kwargs_command:
                    raise ValueError("Cannot specify both command and calc_exec")

                kwargs_command["command"] = f"{calculator_exec} -in PREFIX.pwi > PREFIX.pwo"

            # command or env var must be set
            assert "command" in kwargs_command or "ASE_ESPRESSO_CALCULATOR" in os.environ

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
            if "_output_prefix" in atoms.info:
                save_results(atoms, properties, atoms.info["_output_prefix"])
                atoms.info["_results_saved"] = True
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
