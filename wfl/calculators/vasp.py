"""
VASP calculator

"""

import os
import pathlib
import tempfile
import warnings
import json

import numpy as np

from ase.atoms import Atoms
import ase.calculators.vasp.vasp

from .utils import clean_rundir, handle_nonperiodic, save_results

# NOMAD compatible, see https://nomad-lab.eu/prod/rae/gui/uploads
__default_keep_files = ["POSCAR", "INCAR", "KPOINTS", "OUTCAR", "vasprun.xml", "vasp.out"]
__default_properties = ["energy", "forces", "stress"]


class Vasp(ase.calculators.vasp.vasp.Vasp):
    """Extension of ASE's Vasp calculator that can be used by wfl.calculators.generic

    Parameters
    ----------
    workdir_root: path-like, default os.getcwd()
        directory to put calculation directories into
    dir_prefix: str, default 'VASP_run\_'
        directory name prefix for calculations
    keep_files: bool / None / "default" / list(str), default "default"
        what kind of files to keep from the run
            True : everything kept
            None, False : nothing kept
            "default"   : only ones needed for NOMAD uploads ('POSCAR', 'INCAR', 'KPOINTS', 'OUTCAR', 'vasprun.xyz')
            list(str)   : list of file globs to save
    calculator_command: str
        command to run for vasp (overrides VASP_COMMAND and VASP_COMMAND_GAMMA)

    **kwargs: arguments for ase.calculators.vasp.vasp.Vasp
        May include optional additional keys 'INCAR_file' and 'KPOINTS_file'.  Normal VASP calculator
        keys override contents of INCAR and KPOINTS.  To avoid ASE Vasp's annoying default POTCAR path
        heuristics, key VASP\_PP\_PATH will be used to set corresponding env var (directory above <chem_symbol>/POTCAR),
        and if 'pp' (extra path below VASP\_PP\_PATH) is not specified it will default to '.', rather than to guess
        based on XC.
    """

    implemented_properties = ["energy", "forces", "stress", "free_energy", "magmom", "magmoms"]

    default_parameters = {}

    # default value of wfl_num_inputs_per_python_subprocess for calculators.generic,
    # to override that function's built-in default of 10
    wfl_generic_num_inputs_per_python_subprocess = 1


    def __init__(self, atoms=None, keep_files="default", dir_prefix="run_VASP_",
                 calculator_command=None, **kwargs):
        super(Vasp, self).__init__(**kwargs):

        self.keep_files = keep_files
        self.dir_prefix = dir_prefix
        self.workdir_root = workdir_root
        self.workdir_root.mkdir(parents=True, exist_ok=True)

        if calculator_command is not None:
            self.parameters["command"] = calculator_command

        self.initial_parameters = deepcopy(self.parameters)


    def calculate(self, atoms=None, properties=__default_properties, system_changes=all_changes):
        """Do the calculation. Handles the working directories in addition to regular 
        ASE calculation operations (writing input, executing, reading_results)"""

        if atoms is not None:
            self.atoms = atoms.copy()

        # hack to disable parallel I/O, since that makes VASP's I/O break if
        # this script is being run with mpirun
        from ase.parallel import world, DummyMPI

        orig_world_comm = world.comm
        world.comm = DummyMPI()

        # override Vasp's annoying PAW path heuristics
        VASP_PP_PATH = kwargs.pop("VASP_PP_PATH", None)
        if VASP_PP_PATH is not None:
            os.environ["VASP_PP_PATH"] = VASP_PP_PATH
        if "pp" not in kwargs:
            kwargs["pp"] = "."

        vasp_kwargs_def = {
            "isif": 2,
            "isym": 0,
            "nelm": 300,
            "ediff": 1.0e-7,
            "ismear": 0,
            "sigma": 0.05,
            "lwave": False,
            "lcharg": False,
        }
        vasp_kwargs_def.update(calculator_kwargs)
        calculator_kwargs = vasp_kwargs_def
        if 'WFL_VASP_KWARGS' in os.environ:
            try:
                env_kwargs = json.loads(os.environ['WFL_VASP_KWARGS'])
            except:
                with open(os.environ['WFL_VASP_KWARGS']) as fin:
                    env_kwargs = json.load(fin)
            calculator_kwargs.update(env_kwargs)

        if workdir_root is None:
            # using the current directory
            workdir_root = os.getcwd()
        else:
            pathlib.Path(workdir_root).mkdir(parents=True, exist_ok=True)

        # VASP requires periodic cells with non-zero cell vectors
        assert atoms.get_volume() > 0.0

        nonperiodic, properties_use = handle_nonperiodic(at, properties)
        # VASP cannot handle nonperiodic, and current Vasp calculator complains if pbc=F
        orig_pbc = atoms.pbc.copy()
        atoms.pbc = [True] * 3
        # VASP requires positive triple scalar product - maybe we should just fix this?
        assert atoms.get_volume() > 0.0

        rundir = tempfile.mkdtemp(dir=workdir_root, prefix=dir_prefix)

        # remove parameters that ase.calculators.vasp.Vasp doesn't know about
        incar_file = calculator_kwargs.pop("INCAR_file", None)
        kpoints_file = calculator_kwargs.pop("KPOINTS_file", None)

        # create calc
        atoms.calc = Vasp(directory=rundir, **calculator_kwargs)
        if calculator_command is None and nonperiodic:
            calculator_command = os.environ.get("VASP_COMMAND_GAMMA", None)
        if calculator_command is None:
            assert "VASP_COMMAND" in os.environ
        atoms.calc.command = calculator_command

        # read from INCAR, KPOINTS if provided
        if incar_file is not None:
            atoms.calc.read_incar(incar_file)
        if nonperiodic:
            calculator_kwargs["kspacing"] = 100000.0
            calculator_kwargs["kgamma"] = True
        elif kpoints_file is not None:
            atoms.calc.read_kpoints(kpoints_file)
        # override with original kwargs
        atoms.calc.set(**calculator_kwargs)

        atoms.info["vasp_rundir"] = rundir

        # should we require anything else?
        if atoms.calc.float_params["encut"] is None:
            raise RuntimeError("Refusing to run without explicit ENCUT")

        try:
            super().calculate(atoms=atoms, properties=properties, system_changes=system_changes)
            calculation_succeeded = True
            if 'DFT_FAILED_VASP' in atoms.info:
                del atoms.info['DFT_FAILED_VASP']
        except Exception as exc:
            atoms.info['DFT_FAILED_VASP'] = True
            calculation_succceeded = False
            raise exc
        finally:
            clean_rundir(rundir, self.keep_files, __default_keep_files, calculation_succeeded)

            self.parameters = deepcopy(self.initial_parameters)

            # must reset pbcs here, because save_results will trigger a pbc check
            # inside Vasp.calculate, which will otherwise fail for nonperiodic systems
            atoms.pbc[:] = orig_pbc

            world.comm = orig_world_comm
