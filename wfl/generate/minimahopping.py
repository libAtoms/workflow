import os
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
from ase.optimize.minimahopping import MinimaHopping
from ase.io.trajectory import Trajectory
import ase.io

from wfl.autoparallelize import autoparallelize, autoparallelize_docstring
from wfl.utils.misc import atoms_to_list
from wfl.generate.utils import config_type_append
from wfl.utils.parallel import construct_calculator_picklesafe



def _get_MD_trajectory(rundir):

    md_traj = []
    mdtrajfiles = sorted([file for file in Path(rundir).glob("md*.traj")])
    for mdtraj in mdtrajfiles:
        for at in ase.io.read(f"{mdtraj}", ":"):
            config_type_append(at, 'traj')
            md_traj.append(at)

    return md_traj


# perform MinimaHopping on one ASE.atoms object
def _atom_opt_hopping(atom, calculator, Ediff0, T0, minima_threshold, mdmin,
                     fmax, timestep, totalsteps, skip_failures, **opt_kwargs):
    save_tmpdir = opt_kwargs.pop("save_tmpdir", False)
    return_all_traj = opt_kwargs.pop("return_all_traj", False)
    workdir = os.getcwd()

    rundir = tempfile.mkdtemp(dir=workdir, prefix='Opt_hopping_')

    os.chdir(rundir)
    atom.calc = calculator
    try:
        opt = MinimaHopping(atom, Ediff0=Ediff0, T0=T0, minima_threshold=minima_threshold,
                            mdmin=mdmin, fmax=fmax, timestep=timestep, **opt_kwargs)
        opt(totalsteps=totalsteps)
    except Exception as exc:
        # optimization may sometimes fail to converge.
        if skip_failures:
            sys.stderr.write(f'Structure optimization failed with exception \'{exc}\'\n')
            sys.stderr.flush()
            os.chdir(workdir)
            shutil.rmtree(rundir)
            return None
        else:
            raise
    else:
        traj = []
        if return_all_traj:
            traj += _get_MD_trajectory(rundir)

        for hop_traj in Trajectory('minima.traj'):
            config_type_append(hop_traj, 'minima')
            traj.append(hop_traj)
        os.chdir(workdir)
        if not save_tmpdir:
            shutil.rmtree(rundir)
        return traj


def _run_autopara_wrappable(atoms, calculator, Ediff0=1, T0=1000, minima_threshold=0.5, mdmin=2,
                           fmax=1, timestep=1, totalsteps=10, skip_failures=True,
                           autopara_rng_seed=None, autopara_per_item_info=None,
                           **opt_kwargs):
    """runs a structure optimization

    Parameters
    ----------
    atoms: list(Atoms)
        input configs
    calculator: Calculator / (initializer, args, kwargs)
        ASE calculator or routine to call to create calculator
    Ediff0: float, default 1 (eV)
        initial energy acceptance threshold
    T0: float, default 1000 (K)
        initial MD ‘temperature’
    minima_threshold: float, default 0.5 (Å)
        threshold for identical configs
    mdmin: int, default 2
        criteria to stop MD simulation (number of minima)
    fmax: float, default 1 (eV/Å)
        max force for optimizations
    timestep: float, default 1 (fs)
        timestep for MD simulations
    totalsteps: int, default 10
        number of steps
    skip_failures: bool, default True
        just skip optimizations that raise an exception
    opt_kwargs
        keyword arguments for MinimaHopping
    autopara_rng_seed: int, default None
        global seed used to initialize rng so that each operation uses a different but
        deterministic local seed, use a random value if None

    Returns
    -------
        list(Atoms) trajectories
    """

    calculator = construct_calculator_picklesafe(calculator)
    all_trajs = []

    for at_i, at in enumerate(atoms_to_list(atoms)):
        if autopara_per_item_info is not None:
            np.random.seed(autopara_per_item_info[at_i]["rng_seed"])

        traj = _atom_opt_hopping(atom=at, calculator=calculator, Ediff0=Ediff0, T0=T0, minima_threshold=minima_threshold,
                                 mdmin=mdmin, fmax=fmax, timestep=timestep, totalsteps=totalsteps,
                                 skip_failures=skip_failures, **opt_kwargs)
        all_trajs.append(traj)

    return all_trajs


# run that operation on ConfigSet, for multiprocessing
def minimahopping(*args, **kwargs):
    default_autopara_info = {"num_inputs_per_python_subprocess": 10}

    return autoparallelize(_run_autopara_wrappable, *args,
                           default_autopara_info=default_autopara_info, **kwargs)
autoparallelize_docstring(minimahopping, _run_autopara_wrappable, "Atoms")
