"""
Quantum Espresso interface
"""

import os
import pathlib
import tempfile
import subprocess
import warnings

import numpy as np
from ase import Atoms
from ase.calculators.calculator import all_changes, CalculationFailed
from ase.calculators.espresso import Espresso
try:
    from ase.calculators.espresso import EspressoProfile
except ImportError:
    EspressoProfile = None
from ase.io.espresso import kspacing_to_grid

from .utils import clean_rundir, handle_nonperiodic, save_results
from ..utils.misc import atoms_to_list

# NOMAD compatible, see https://nomad-lab.eu/prod/rae/gui/uploads
__default_keep_files = ["*.pwo"]
__default_properties = ["energy", "forces", "stress"]


def evaluate_op(
    atoms,
    base_rundir=None,
    dir_prefix="run_QE_",
    calculator_command=None,
    calculator_kwargs=None,
    output_prefix="QE_",
    properties=None,
    keep_files="default",
):
    """Evaluate a configuration with Quantum Espresso

    Parameters
    ----------
    atoms: Atoms / list(Atoms)
        input atomic configs
    base_rundir: path-like, default os.getcwd()
        directory to put calculation directories into
    dir_prefix: str, default 'QE-run\_'
        directory name prefix for calculations
    calculator_command: str
        command for QE, without any prefix or redirection set.
        for example: "mpirun -n 4 /path/to/pw.x"
    calculator_kwargs : dict
    output_prefix : str / None, default 'QE\_'
        prefix for info/arrays keys, None for SinglePointCalculator
    properties : list(str), default None
        ase-compatible property names, None for default list (energy, forces, stress)
    keep_files: bool / None / "default" / list(str), default "default"
        what kind of files to keep from the run
            - True : everything kept
            - None, False : nothing kept, unless calculation fails
            - "default"   : only ones needed for NOMAD uploads ('\*.pwo')
            - list(str)   : list of file globs to save

    Returns
    -------
        Atoms or list(Atoms) with calculated properties
    """
    # use list of atoms in any case
    at_list = atoms_to_list(atoms)

    # default properties
    if properties is None:
        properties = __default_properties

    # keyword setup
    if calculator_kwargs is None:
        raise ValueError("QE will not perform a calculation without settings given!")

    if base_rundir is None:
        # using the current directory
        base_rundir = os.getcwd()
    else:
        pathlib.Path(base_rundir).mkdir(parents=True, exist_ok=True)

    for at in at_list:

        # periodicity
        kwargs_this_calc, properties_use = qe_kpoints_and_kwargs(
            at, calculator_kwargs, properties
        )

        if calculator_command is not None:
            if EspressoProfile is None:
                # older syntax
                kwargs_this_calc["command"] = f"{calculator_command} -in PREFIX.pwi > PREFIX.pwo"
            else:
                # newer syntax
                kwargs_this_calc['profile'] = EspressoProfile(argv=calculator_command.split())

        # create temp dir and calculator
        rundir = tempfile.mkdtemp(dir=base_rundir, prefix=dir_prefix)
        at.calc = Espresso(directory=rundir, **kwargs_this_calc)

        # calculate
        calculation_succeeded = False
        try:
            at.calc.calculate(at, properties=properties_use, system_changes=all_changes)
            calculation_succeeded = True
        except Exception as exc:
            # CalculationFailed is probably what's supposed to be returned
            # for failed convergence, Espresso currently returns subprocess.CalledProcessError
            #     since pw.x returns a non-zero status
            warnings.warn(f'Calculation failed with exc {exc}')

        # save results
        if calculation_succeeded:
            save_results(at, properties_use, output_prefix)

        # clean run directory
        clean_rundir(rundir, keep_files, __default_keep_files, calculation_succeeded)

    if isinstance(atoms, Atoms):
        return at_list[0]
    else:
        return at_list


def qe_kpoints_and_kwargs(
    atoms: Atoms, kwargs: dict, properties: list,
):
    """Handle K-Points in QE for any periodicity

    - stress is only calculated if all directions are periodic
    - gamma point is used if none of the directions are periodic
    - single K-Point is used in any direction which is not periodic

    Parameters
    ----------
    atoms: Atoms
    kwargs: dict
        QE calculator's intended keyword arguments
    properties: list
        ASE-compatible property name list for calculation

    Returns
    -------
    modified_kwargs: dict
    properties_use: list

    """
    # periodicity, allowing mixed periodicity
    nonperiodic, properties_use = handle_nonperiodic(
        atoms, properties, allow_mixed=True
    )

    # a copy of the parameters that we are modifying for each calculation
    modified_kwargs = dict(kwargs)

    # stress and force calculations need keys
    modified_kwargs["tprnfor"] = "forces" in properties_use
    modified_kwargs["tstress"] = "stress" in properties_use

    if nonperiodic:
        if not np.any(atoms.get_pbc()):
            # FFF -> gamma point only
            modified_kwargs["kpts"] = None
            modified_kwargs["kspacing"] = None
            modified_kwargs["koffset"] = False
        else:
            # mixed T & F
            if "kspacing" in kwargs:
                # need to create the grid,
                # `kspacing` overwrites `kpts`
                # we set it in there and
                modified_kwargs["kpts"] = kspacing_to_grid(
                    atoms, spacing=kwargs["kspacing"] / (2.0 * np.pi)
                )

            # kspacing None anyways
            modified_kwargs["kspacing"] = None

            # original, or overwritten from kspacing
            if "kpts" in modified_kwargs:
                # any no-periodic direction has 1 k-point only
                kpts = np.array(modified_kwargs["kpts"])
                kpts[~atoms.get_pbc()] = 1
                modified_kwargs["kpts"] = tuple(kpts)

            # k-point offset
            k_offset = kwargs.get("koffset", False)
            if k_offset is True:
                k_offset = (1, 1, 1)

            if k_offset:
                # set to zero on any non-periodic ones
                k_offset = np.array(k_offset)
                k_offset[~atoms.get_pbc()] = 0
                modified_kwargs["koffset"] = tuple(k_offset)
            else:
                modified_kwargs["koffset"] = k_offset

    return modified_kwargs, properties_use
