"""
VASP calculator

"""

import os
from pathlib import Path

import json
from copy import deepcopy

import numpy as np

from ase.atoms import Atoms
from ase.calculators.calculator import all_changes
from ase.calculators.vasp.vasp import Vasp as ASE_Vasp

from .wfl_fileio_calculator import WFLFileIOCalculator
from .utils import handle_nonperiodic

# NOMAD compatible, see https://nomad-lab.eu/prod/rae/gui/uploads
_default_keep_files = ["POSCAR", "INCAR", "KPOINTS", "OUTCAR", "vasprun.xml", "vasp.out"]
_default_properties = ["energy", "forces", "stress"]

class Vasp(WFLFileIOCalculator, ASE_Vasp):
    """Extension of ASE's Vasp calculator that can be used by wfl.calculators.generic

    Notes
    -----
    "directory" argument cannot be present. Use rundir and workdir instead.
    "pp" defaults to ".", so VASP_PP_PATH env var is absolute path to "<elem name>/POTCAR" files

    Parameters
    ----------
    keep_files: bool / None / "default" / list(str), default "default"
        what kind of files to keep from the run
            True, "*" : everything kept
            None, False : nothing kept
            "default"   : only ones needed for NOMAD uploads ('POSCAR', 'INCAR', 'KPOINTS', 'OUTCAR', 'vasprun.xml', 'vasp.out')
            list(str)   : list of file globs to save
    rundir: str / Path, default 'run\_VASP\_'
        Run directory name prefix (or full name - see reuse_rundir)
    reuse_rundir: bool, default False
        Treat rundir as a fixed directory, rather than a prefix to a unique dir name.
        WARNING: Do not set rundir to an existing directory with other files, because they
        may be deleted by clean_rundir() at the end of the calculation.
    workdir: str / Path, default . at calculate time
        Path in which rundir will be created.
    scratchdir: str / Path, default None
        temporary directory to execute calculations in and delete or copy back results (set by
        "keep_files") if needed.  For example, directory on a local disk with fast file I/O.
    calculator_command: str
        command to run for vasp (overrides ASE_VASP_COMMAND and ASE_VASP_COMMAND_GAMMA)
    **kwargs: arguments for ase.calculators.vasp.vasp.Vasp
        remaining arguments to ASE's Vasp calculator constructor
    """

    # default value of wfl_num_inputs_per_python_subprocess for calculators.generic,
    # to override that function's built-in default of 10
    wfl_generic_num_inputs_per_python_subprocess = 1

    # note that we also have a default for "pp", but that has to be handlded separately
    default_parameters = ASE_Vasp.default_parameters.copy()
    default_parameters.update({
        "isif": 2,
        "isym": 0,
        "nelm": 300,
        "ismear": 0,
        "sigma": 0.05,
        "ediff": 1.0e-7,
        "lwave": False,
        "lcharg": False,
    })

    def __init__(self, keep_files="default", rundir="run_VASP_", reuse_rundir=False,
                 workdir=".", scratchdir=None, calculator_command=None, **kwargs):

        # get initialparams from env var
        kwargs_use = {}
        if 'WFL_VASP_KWARGS' in os.environ:
            try:
                kwargs_use = json.loads(os.environ['WFL_VASP_KWARGS'])
            except json.decoder.JSONDecodeError:
                with open(os.environ['WFL_VASP_KWARGS']) as fin:
                    kwargs_use = json.load(fin)

        # override with explicitly passed in values
        kwargs_use.update(kwargs)

        # pp is not handled by Vasp.default_parameters, because that only includes
        # parameters that are in INCAR
        if "pp" not in kwargs_use:
            kwargs_use["pp"] = "."

        if calculator_command is not None:
            if "command" in kwargs_use:
                raise ValueError("Got calculator_command and command arguments")
            kwargs_use["command"] = calculator_command

        # WFLFileIOCalculator is a mixin, will call remaining superclass constructors for us
        super().__init__(keep_files=keep_files, rundir=rundir, reuse_rundir=reuse_rundir,
                         workdir=workdir, scratchdir=scratchdir, **kwargs_use)


    def calculate(self, atoms=None, properties=_default_properties, system_changes=all_changes):
        """Do the calculation. Handles the working directories in addition to regular 
        ASE calculation operations (writing input, executing, reading_results)"""

        if atoms is not None:
            self.atoms = atoms.copy()

        # hack to disable parallel I/O, since that makes VASP's I/O break if
        # this script is being run with mpirun
        from ase.parallel import world, DummyMPI

        orig_world_comm = world.comm
        world.comm = DummyMPI()

        orig_VASP_PP_PATH = os.environ.get("VASP_PP_PATH")
        if orig_VASP_PP_PATH is None:
            if self.input_params['pp'].startswith("/"):
                os.environ["VASP_PP_PATH"] = "/."
            else:
                os.environ["VASP_PP_PATH"] = "."

        # VASP requires periodic cells with non-zero cell vectors
        if atoms.get_volume() < 0.0:
            (atoms.cell[0], atoms.cell[1]) = (atoms.cell[1], atoms.cell[0])
            permuted_a0_a1 = True
        else:
            permuted_a0_a1 = False

        nonperiodic, properties_use = handle_nonperiodic(atoms, properties)
        # VASP cannot handle nonperiodic, and current Vasp calculator complains if pbc=F
        orig_pbc = atoms.pbc.copy()
        atoms.pbc = [True] * 3

        self.setup_rundir()

        orig_command = self.command
        if self.command is None:
            if nonperiodic:
                self.command = os.environ.get("ASE_VASP_COMMAND_GAMMA", None)
            elif "ASE_VASP_COMMAND" not in os.environ:
                raise RuntimeError("Need env var ASE_VASP_COMMAND for periodic systems if command is not explicitly passed to constructor")

        # set some things for nonperiodic systems
        orig_kspacing = self.float_params["kspacing"]
        orig_kgamma = self.bool_params["kgamma"]
        if nonperiodic:
            self.float_params["kspacing"] = 100000.0
            self.bool_params["kgamma"] = True

        atoms.info["vasp_rundir"] = str(self._cur_rundir)

        if self.float_params["encut"] is None:
            raise RuntimeError("Refusing to run without explicit ENCUT")
        # should we require anything except ENCUT?

        calculation_succceeded = False
        try:
            super().calculate(atoms=atoms, properties=properties_use, system_changes=system_changes)
            calculation_succeeded = True
            if 'DFT_FAILED_VASP' in atoms.info:
                del atoms.info['DFT_FAILED_VASP']
        except Exception as exc:
            atoms.info['DFT_FAILED_VASP'] = True
            raise exc
        finally:
            self.clean_run(_default_keep_files, calculation_succeeded)

            # undo pbc change
            atoms.pbc[:] = orig_pbc

            # undo cell vector permutation
            if permuted_a0_a1:
                (atoms.cell[0], atoms.cell[1]) = (atoms.cell[1], atoms.cell[0])

            # undo nonperiodic related changes
            self.command = orig_command
            self.float_params["kspacing"] = orig_kspacing
            self.bool_params["kgamma"] = orig_kgamma

            # undo env VASP_PP_PATH
            if orig_VASP_PP_PATH is not None:
                os.environ["VASP_PP_PATH"] = orig_VASP_PP_PATH

            # undo communicator mangling
            world.comm = orig_world_comm
