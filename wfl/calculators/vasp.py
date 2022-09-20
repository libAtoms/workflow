"""
VASP calculator

"""

import os
from pathlib import Path
import tempfile
import json
from copy import deepcopy

import numpy as np

from ase.atoms import Atoms
from ase.calculators.calculator import all_changes
import ase.calculators.vasp.vasp

from .utils import clean_rundir, handle_nonperiodic

# NOMAD compatible, see https://nomad-lab.eu/prod/rae/gui/uploads
_default_keep_files = ["POSCAR", "INCAR", "KPOINTS", "OUTCAR", "vasprun.xml", "vasp.out"]
_default_properties = ["energy", "forces", "stress"]


class Vasp(ase.calculators.vasp.vasp.Vasp):
    """Extension of ASE's Vasp calculator that can be used by wfl.calculators.generic

    Parameters
    ----------
    dir_prefix: str, default 'run\_VASP\_'
        directory name prefix for calculations
    keep_files: bool / None / "default" / list(str), default "default"
        what kind of files to keep from the run
            True : everything kept
            None, False : nothing kept
            "default"   : only ones needed for NOMAD uploads ('POSCAR', 'INCAR', 'KPOINTS', 'OUTCAR', 'vasprun.xyz')
            list(str)   : list of file globs to save
    calculator_command: str
        command to run for vasp (overrides ASE_VASP_COMMAND and ASE_VASP_COMMAND_GAMMA)

    **kwargs: arguments for ase.calculators.vasp.vasp.Vasp
        To avoid ASE Vasp's annoying default POTCAR path heuristics, key VASP\_PP\_PATH will be
        used to set corresponding env var (directory above <chem_symbol>/POTCAR), and if 'pp'
        (extra path below VASP\_PP\_PATH) is not specified it will default to '.', rather than
        to guess based on XC.
    """

    # default value of wfl_num_inputs_per_python_subprocess for calculators.generic,
    # to override that function's built-in default of 10
    wfl_generic_num_inputs_per_python_subprocess = 1

    __vasp_kwargs_def = {
        "isif": 2,
        "isym": 0,
        "nelm": 300,
        "ismear": 0,
        "sigma": 0.05,
        "ediff": 1.0e-7,
        "lwave": False,
        "lcharg": False,
        "pp": "."
    }

    def __init__(self, atoms=None, keep_files="default",
                 dir_prefix="run_VASP_", workdir_root=None,
                 calculator_command=None, VASP_PP_PATH=None,
                 **kwargs):

        kwargs_use = deepcopy(kwargs)

        # get params from env var if not explicitly passed in
        if 'WFL_VASP_KWARGS' in os.environ:
            try:
                env_kwargs = json.loads(os.environ['WFL_VASP_KWARGS'])
            except:
                with open(os.environ['WFL_VASP_KWARGS']) as fin:
                    env_kwargs = json.load(fin)
            for k, v in env_kwargs.items():
                if k not in kwargs_use:
                    kwargs_use[k] = v

        # get params from our defaults if not set yet
        for k, v in self.__vasp_kwargs_def.items():
            if k not in kwargs_use:
                kwargs_use[k] = v

        if calculator_command is not None:
            if "command" in kwargs_use:
                raise ValueError("Got calculator_command and command arguments")
            kwargs_use["command"] = calculator_command

        super(Vasp, self).__init__(**kwargs_use)

        self._keep_files = keep_files
        self._dir_prefix = dir_prefix
        self._workdir_root = Path(workdir_root) if workdir_root is not None else None
        self._override_VASP_PP_PATH = VASP_PP_PATH


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
        if self._override_VASP_PP_PATH is not None:
            os.environ["VASP_PP_PATH"] = str(self._override_VASP_PP_PATH)

        if self._workdir_root is None:
            workdir_root = Path.cwd()
        else:
            workdir_root = self._workdir_root

        workdir_root.mkdir(parents=True, exist_ok=True)

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

        self.directory = Path(tempfile.mkdtemp(dir=workdir_root, prefix=self._dir_prefix))

        # create calc
        orig_command = self.command
        if self.command is None:
            if nonperiodic:
                self.command = os.environ.get("ASE_VASP_COMMAND_GAMMA", None)
            elif "ASE_VASP_COMMAND" not in os.environ:
                raise RuntimeError("Need env var ASE_VASP_COMMAND for periodic systems if command is not explicitly passed to constructor")

        # read from INCAR, KPOINTS if provided
        if nonperiodic:
            orig_kspacing = self.float_params["kspacing"]
            orig_kgamma = self.bool_params["kgamma"]
            self.float_params["kspacing"] = 100000.0
            self.bool_params["kgamma"] = True

        atoms.info["vasp_rundir"] = str(self.directory)

        # should we require anything else?
        if self.float_params["encut"] is None:
            raise RuntimeError("Refusing to run without explicit ENCUT")

        try:
            super().calculate(atoms=atoms, properties=properties_use, system_changes=system_changes)
            calculation_succeeded = True
            if 'DFT_FAILED_VASP' in atoms.info:
                del atoms.info['DFT_FAILED_VASP']
        except Exception as exc:
            atoms.info['DFT_FAILED_VASP'] = True
            calculation_succceeded = False
            raise exc
        finally:
            clean_rundir(self.directory, self._keep_files, _default_keep_files, calculation_succeeded)

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
