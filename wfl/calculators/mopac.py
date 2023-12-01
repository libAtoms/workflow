"""
MOPAC interface
"""
from ase.calculators.mopac import MOPAC as ASE_MOPAC
from ase.calculators.calculator import all_changes

from .wfl_fileio_calculator import WFLFileIOCalculator

_default_keep_files = ["*.out"]
_default_properties = ["energy", "forces"]

class MOPAC(WFLFileIOCalculator, ASE_MOPAC):
    """Extension of ASE's MOPAC claculator so that it can be used by wfl.calculators.generic (mainly each
    calculation is run in a separate directory)
    """

    wfl_generic_default_autopara_info = {"num_inputs_per_python_subprocess": 1}

    def __init__(self, keep_files="default", rundir_prefix="run_MOPAC_",
                 workdir=None, scratchdir=None,
                 calculator_exec=None, **kwargs):

        # WFLFileIOCalculator is a mixin, will call remaining superclass constructors for us
        super().__init__(keep_files=keep_files, rundir_prefix=rundir_prefix,
                         workdir=workdir, scratchdir=scratchdir, **kwargs)

    def calculate(self, atoms=None, properties=_default_properties, system_changes=all_changes):
        """Do the calculation. Handles the working directories in addition to regular
        ASE calculation operations (writing input, executing, reading_results)
        Reimplements & extends GenericFileIOCalculator.calculate() for the development version of ASE
        or FileIOCalculator.calculate() for the v3.22.1"""

        # from WFLFileIOCalculator
        self.setup_rundir()

        try:
            super().calculate(atoms=atoms, properties=properties, system_changes=system_changes)
            calculation_succeeded = True
            if 'FAILED_MOPAC' in atoms.info:
                del atoms.info['FAILED_MOPAC']
        except Exception as exc:
            atoms.info['FAILED_MOPAC'] = True
            calculation_succeeded = False
            raise exc
        finally:
            # from WFLFileIOCalculator
            self.clean_rundir(_default_keep_files, calculation_succeeded)
