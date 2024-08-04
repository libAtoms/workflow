"""
FHI-Aims Calculator
"""

from copy import deepcopy
import numpy as np

from ase.calculators.calculator import all_changes
from ase.calculators.aims import Aims as ASE_Aims
try:
    from ase.calculators.aims import AimsProfile
except ImportError:
    AimsProfile = None

from .wfl_fileio_calculator import WFLFileIOCalculator

# NOMAD compatible, see https://nomad-lab.eu/prod/rae/gui/uploads
_default_keep_files = ["control.in", "geometry.in", "aims.out"]
_default_properties = ["energy", "forces", "stress"]


class Aims(WFLFileIOCalculator, ASE_Aims):
    """Extension of ASE's Aims calculator that can be used by wfl.calculators.generic.

    The ```directory``` argument from the basic calculator implementation in ASE cannot be present.
    Use ```rundir_prefix``` and ```workdir``` instead.

    Note that in case the Calculator detects a structure that is non-periodic invalid keywords
    will automatically be removed. Specifically, all keywords that start with 'k_', contain the sub-string
    'k_grid' or 'stress' or are equal to 'relax_unit_cell' or 'external_pressure' will be removed.

    Parameters
    ----------
    keep_files: bool / None / "default" / list(str), default "default"
        What kind of files to keep from the run:
            - True :        Everything kept.
            - None, False : Nothing kept, unless calculation fails.
            - "default" :   Only ones needed for NOMAD uploads ('control.in', 'geometry.in', 'aims.out') are kept.
            - list(str) :   List of file globs to save.
    rundir_prefix: str / Path, default 'run\_Aims\_'
        Run directory name prefix.
    workdir: str / Path, default . at calculate time
        Path in which the run directory will be created.
    scratchdir: str / Path, default None
        Temporary directory to execute calculations in and delete or copy back results (set by ```keep_files```)
        if needed.  For example, directory on a local disk with fast file I/O.

    **kwargs: arguments for ase.calculators.aims.Aims
        See https://wiki.fysik.dtu.dk/ase/_modules/ase/calculators/aims.html.
    """
    implemented_properties = ["energy", "forces", "stress"]

    # new default value of num_inputs_per_python_subprocess for calculators.generic,
    # to override that function's built-in default of 10
    wfl_generic_default_autopara_info = {"num_inputs_per_python_subprocess": 1}

    def __init__(self, keep_files="default", rundir_prefix="run_Aims_", workdir=None,
                 scratchdir=None, **kwargs):

        kwargs_command = deepcopy(kwargs)

        # WFLFileIOCalculator is a mixin, will call remaining superclass constructors for us
        super().__init__(keep_files=keep_files, rundir_prefix=rundir_prefix,
                         workdir=workdir, scratchdir=scratchdir, **kwargs_command)

        # we modify the parameters in self.calculate() based on the individual atoms object,
        # so let's make a copy of the initial parameters
        self.initial_parameters = deepcopy(self.parameters)

    def calculate(self, atoms=None, properties=_default_properties, system_changes=all_changes):
        """Do the calculation.

        Handles the working directories in addition to regular ASE calculation operations (writing input, executing,
        reading_results). Reimplements and extends GenericFileIOCalculator.calculate() for the development version
        of ASE or FileIOCalculator.calculate() for the v3.22.1.

        Parameter:
        ----------
        atoms: ASE-Atoms object
            Atomistic system to be calculated.
        properties: list(str)
            List of what needs to be calculated.  Can be any combination of 'energy',
            'forces', 'stress', 'dipole', 'charges', 'magmom' and 'magmoms'.
        system_changes: list(str)
            List of what has changed since last calculation.  Can be any combination of
            these six: 'positions', 'numbers', 'cell', 'pbc', 'initial_charges' and 'initial_magmoms'.
        """
        if atoms is not None:
            self.atoms = atoms.copy()

        # this may modify self.parameters, will be reset to the initial ones after calculation
        self._setup_calc_params()

        # from WFLFileIOCalculator
        self.setup_rundir()

        try:
            super().calculate(atoms=atoms, properties=properties, system_changes=system_changes)
            calculation_succeeded = True
            if 'DFT_FAILED_AIMS' in atoms.info:
                del atoms.info['DFT_FAILED_AIMS']
        except Exception as exc:
            atoms.info['DFT_FAILED_AIMS'] = True
            calculation_succeeded = False
            raise exc
        finally:
            # from WFLFileIOCalculator
            self.clean_rundir(_default_keep_files, calculation_succeeded)

            # Reset parameters to what they were when the calculator was initialized.
            self.parameters = deepcopy(self.initial_parameters)

    def _setup_calc_params(self):
        """
        Setup parameters for the calculation based on the atomistic systems periodicity.

        Remove entries in ```self.parameters``` that don't fit to the atomistic systems periodicity.
        """
        if np.all(~self.atoms.pbc):  # PBC is FFF
            # e.g. k_grid or k_grid_density, k_offset, sc_accuracy_stress
            rm_parameters = [param_i for param_i in self.parameters if 'k_grid' in param_i
                             or param_i.startswith('k_') or 'stress' in param_i
                             or param_i in ['relax_unit_cell', 'external_pressure']]
            for param_i in rm_parameters:
                self.parameters.pop(param_i)



