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
from ase.calculators.vasp.vasp import Vasp

from .utils import clean_rundir, handle_nonperiodic, save_results
from ..utils.misc import atoms_to_list

# NOMAD compatible, see https://nomad-lab.eu/prod/rae/gui/uploads
__default_keep_files = ["POSCAR", "INCAR", "KPOINTS", "OUTCAR", "vasprun.xml", "vasp.out"]
__default_properties = ["energy", "forces", "stress"]


def evaluate_op(
    atoms,
    base_rundir=None,
    dir_prefix="run_VASP_",
    calculator_command=None,
    calculator_kwargs=None,
    output_prefix="VASP_",
    properties=None,
    keep_files="default",
):
    """Evaluate a configuration with VASP

    Parameters
    ----------
    atoms: Atoms / list(Atoms)
        input atomic configs
    base_rundir: path-like, default os.getcwd()
        directory to put calculation directories into
    dir_prefix: str, default 'VASP_run\_'
        directory name prefix for calculations
    calculator_command: str
        command to run for vasp (overrides VASP_COMMAND and VASP_COMMAND_GAMMA)
    calculator_kwargs: dict
        arguments to Vasp, plus optional additional keys 'INCAR_file' and 'KPOINTS_file'.  Normal VASP calculator
        keys override contents of INCAR and KPOINTS.  To avoid ASE Vasp's annoying default POTCAR path
        heuristics, key VASP\_PP\_PATH will be used to set corresponding env var (directory above <chem_symbol>/POTCAR),
        and if 'pp' (extra path below VASP\_PP\_PATH) is not specified it will default to '.', rather than to guess
        based on XC.
    output_prefix: str / None, default 'VASP\_'
        prefix for info/arrays keys where results will be saved, None for SinglePointCalculator
    properties: list(str), default ['energy', 'forces', 'stress']
        properties to calculate.  Note that 'energy' is used to calculate force-consistent
        value (ASE's 'free_energy')
    keep_files: bool / None / "default" / list(str), default "default"
        what kind of files to keep from the run
            True : everything kept
            None, False : nothing kept
            "default"   : only ones needed for NOMAD uploads ('POSCAR', 'INCAR', 'KPOINTS', 'OUTCAR', 'vasprun.xyz')
            list(str)   : list of file globs to save

    Returns
    -------
        Atoms or list(Atoms) with calculated properties
    """

    # hack to disable parallel I/O, since that makes VASP's I/O break if
    # this script is being run with mpirun
    from ase.parallel import world, DummyMPI

    orig_world_comm = world.comm
    world.comm = DummyMPI()

    # use list of atoms in any case
    at_list = atoms_to_list(atoms)

    # default properties
    if properties is None:
        properties = __default_properties

    # default arguments
    if calculator_kwargs is None:
        calculator_kwargs = {}

    # override Vasp's annoying PAW path heuristics
    VASP_PP_PATH = calculator_kwargs.pop("VASP_PP_PATH", None)
    if VASP_PP_PATH is not None:
        os.environ["VASP_PP_PATH"] = VASP_PP_PATH
    if "pp" not in calculator_kwargs:
        calculator_kwargs["pp"] = "."

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

    if base_rundir is None:
        # using the current directory
        base_rundir = os.getcwd()
    else:
        pathlib.Path(base_rundir).mkdir(parents=True, exist_ok=True)

    for at in at_list:
        # VASP requires periodic cells with non-zero cell vectors
        assert at.get_volume() > 0.0

        nonperiodic, properties_use = handle_nonperiodic(at, properties)
        # VASP cannot handle nonperiodic, and current Vasp calculator complains if pbc=F
        orig_pbc = at.pbc.copy()
        at.pbc = [True] * 3
        # VASP requires positive triple scalar product - maybe we should just fix this?
        assert at.get_volume() > 0.0

        rundir = tempfile.mkdtemp(dir=base_rundir, prefix=dir_prefix)

        # remove parameters that ase.calculators.vasp.Vasp doesn't know about
        incar_file = calculator_kwargs.pop("INCAR_file", None)
        kpoints_file = calculator_kwargs.pop("KPOINTS_file", None)

        # create calc
        at.calc = Vasp(directory=rundir, **calculator_kwargs)
        if calculator_command is None and nonperiodic:
            calculator_command = os.environ.get("VASP_COMMAND_GAMMA", None)
        if calculator_command is None:
            assert "VASP_COMMAND" in os.environ
        at.calc.command = calculator_command

        # read from INCAR, KPOINTS if provided
        if incar_file is not None:
            at.calc.read_incar(incar_file)
        if nonperiodic:
            calculator_kwargs["kspacing"] = 100000.0
            calculator_kwargs["kgamma"] = True
        elif kpoints_file is not None:
            at.calc.read_kpoints(kpoints_file)
        # override with original kwargs
        at.calc.set(**calculator_kwargs)

        at.info["vasp_rundir"] = rundir

        # should we require anything else?
        if at.calc.float_params["encut"] is None:
            raise RuntimeError("Refusing to run without explicit ENCUT")

        # calculate
        calculation_succeeded = False
        try:
            at.calc.calculate(at, properties_use)
            calculation_succeeded = True
        except Exception as exc:
            warnings.warn(f"VASP calculation failed with exception {exc}")

        if calculation_succeeded:
            save_results(at, properties_use, output_prefix)

        # must reset pbcs here, because save_results will trigger a pbc check
        # inside Vasp.calculate, which will otherwise fail for nonperiodic systems
        at.pbc[:] = orig_pbc

        clean_rundir(rundir, keep_files, __default_keep_files, calculation_succeeded)

    world.comm = orig_world_comm

    if isinstance(atoms, Atoms):
        return at_list[0]
    else:
        return at_list
