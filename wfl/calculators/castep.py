"""
Quantum Castep interface
"""

from copy import deepcopy

from ase.calculators.calculator import all_changes
from ase.calculators.castep import Castep as ASE_Castep

from .wfl_fileio_calculator import WFLFileIOCalculator

# NOMAD compatible, see https://nomad-lab.eu/prod/rae/gui/uploads
_default_keep_files = ["*.castep", "*.param", "*.cell"]
_default_properties = ["energy", "forces", "stress"]


class Castep(WFLFileIOCalculator, ASE_Castep):
    """Extension of ASE's Castep calculator that can be used by wfl.calculators.generic

    "directory" argument cannot be present. Use rundir_prefix and workdir instead.

    Parameters
    ----------
    keep_files: bool / None / "default" / list(str), default "default"
        what kind of files to keep from the run
            - True : everything kept
            - None, False : nothing kept, unless calculation fails
            - "default"   : only ones needed for NOMAD uploads ('\*.pwo')
            - list(str)   : list of file globs to save
    rundir_prefix: str / Path, default 'run\_CASTEP\_'
        Run directory name prefix
    workdir: str / Path, default . at calculate time
        Path in which rundir will be created.
    scratchdir: str / Path, default None
        temporary directory to execute calculations in and delete or copy back results (set by
        "keep_files") if needed.  For example, directory on a local disk with fast file I/O.
    calculator_exec: str
        command for Castep, without any prefix or redirection set.
        for example: "mpirun -n 4 castep.mpi"
        Alternative for "castep_command", for consistency with other wfl calculators.

    **kwargs: arguments for ase.calculators.Castep.Castep
    """

    implemented_properties = ["energy", "forces", "stress"]

    # new default value of num_inputs_per_python_subprocess for calculators.generic,
    # to override that function's built-in default of 10
    wfl_generic_default_autopara_info = {"num_inputs_per_python_subprocess": 1}

    def __init__(
        self,
        keep_files="default",
        rundir_prefix="run_CASTEP_",
        workdir=None,
        scratchdir=None,
        calculator_exec=None,
        **kwargs,
    ):

        kwargs = deepcopy(kwargs)
        if calculator_exec is not None:
            if "castep_command" in kwargs:
                raise ValueError("Cannot specify both calculator_exec and command")
            kwargs["castep_command"] = calculator_exec

        if kwargs.get("castep_pp_path", None) is None:
            # make sure we are looking for pspot if path given
            kwargs["find_pspots"] = True

        # WFLFileIOCalculator is a mixin, will call remaining superclass constructors for us
        super().__init__(
            keep_files=keep_files,
            rundir_prefix=rundir_prefix,
            workdir=workdir,
            scratchdir=scratchdir,
            **kwargs,
        )

    def calculate(
        self, atoms=None, properties=_default_properties, system_changes=all_changes
    ):
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

        orig_pbc = self.atoms.pbc.copy()
        try:
            super().calculate(
                atoms=atoms, properties=properties, system_changes=system_changes
            )
            calculation_succeeded = True
            if "DFT_FAILED_CASTEP" in atoms.info:
                del atoms.info["DFT_FAILED_CASTEP"]
        except Exception as exc:
            atoms.info["DFT_FAILED_CASTEP"] = True
            calculation_succeeded = False
            raise exc
        finally:
            # ASE castep calculator does not ever raise an exception when
            # it fails.  Instead, you get things like stress being None,
            # which lead to TypeError when save_calc_results calls get_stress().
            for prop in properties:
                result = self.get_property(prop, allow_calculation=False)
                if result is None:
                    calculation_succeeded = False
                    atoms.info["DFT_FAILED_CASTEP"] = True
                    break

            # from WFLFileIOCalculator
            self.clean_rundir(_default_keep_files, calculation_succeeded)

            # reset pbc because Castep overwrites it to True
            self.atoms.pbc = orig_pbc

    def setup_calc_params(self, properties):
        # calculate stress if requested
        self.param.calculate_stress = "stress" in properties
