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
from wfl.utils.save_calc_results import at_copy_save_calc_results
from wfl.utils.misc import atoms_to_list
from .utils import save_config_type
from wfl.utils.parallel import construct_calculator_picklesafe


def _get_MD_trajectory(rundir, update_config_type, prefix):

    md_traj = []
    mdtrajfiles = sorted([file for file in Path(rundir).glob("md*.traj")])
    for mdtraj in mdtrajfiles:
        for at in ase.io.read(f"{mdtraj}", ":"):
            new_config = at_copy_save_calc_results(at, prefix=prefix)
            save_config_type(new_config, update_config_type, 'minhop_traj')
            md_traj.append(new_config)

    return md_traj


# perform MinimaHopping on one ASE.atoms object
def _atom_opt_hopping(atom, calculator, Ediff0, T0, minima_threshold, mdmin,
                     fmax, timestep, totalsteps, skip_failures, update_config_type, results_prefix,
                     workdir=None, **opt_kwargs):
    save_tmpdir = opt_kwargs.pop("save_tmpdir", False)
    return_all_traj = opt_kwargs.pop("return_all_traj", False)
    origdir = Path.cwd()
    if workdir is None:
        workdir = Path.cwd()
    else:
        workdir = Path(workdir)

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
            os.chdir(origdir)
            return None
        else:
            raise
    else:
        traj = []
        if return_all_traj:
            traj += _get_MD_trajectory(rundir, update_config_type, prefix=results_prefix)

        for hop_traj in Trajectory('minima.traj'):
            new_config = at_copy_save_calc_results(hop_traj, prefix=results_prefix)
            save_config_type(new_config, update_config_type, 'minhop_min')
            traj.append(new_config)
        if not save_tmpdir:
            os.chdir(workdir)
            shutil.rmtree(rundir)
        os.chdir(origdir)
        return traj

    os.chdir(origdir)


def _run_autopara_wrappable(atoms, calculator, Ediff0=1, T0=1000, minima_threshold=0.5, mdmin=2,
                           fmax=1, timestep=1, totalsteps=10, skip_failures=True, update_config_type="append",
                           results_prefix='last_op__minhop_', workdir=None, rng=None, _autopara_per_item_info=None,
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
        initial MD temperature
    minima_threshold: float, default 0.5 (A)
        threshold for identical configs
    mdmin: int, default 2
        criteria to stop MD simulation (number of minima)
    fmax: float, default 1 (eV/A)
        max force for optimizations
    timestep: float, default 1 (fs)
        timestep for MD simulations
    totalsteps: int, default 10
        number of steps
    skip_failures: bool, default True
        just skip optimizations that raise an exception
    update_config_type: ["append" | "overwrite" | False], default "append"
        whether/how to add at.info['optimize_config_type'] to at.info['config_type']
    workdir: str/Path default None
        workdir for saving files
    opt_kwargs
        keyword arguments for MinimaHopping
    rng: numpy.random.Generator, default None
        random number generator to use (needed for pressure sampling, initial temperature, or Langevin dynamics)
    _autopara_per_item_info: dict
        INTERNALLY used by autoparallelization framework to make runs reproducible (see
        wfl.autoparallelize.autoparallelize() docs)

    Returns
    -------
        list(Atoms) trajectories
    """

    calculator = construct_calculator_picklesafe(calculator)
    all_trajs = []

    for at_i, at in enumerate(atoms_to_list(atoms)):
        if _autopara_per_item_info is not None:
            # minima hopping doesn't let you pass in a np.random.Generator, so set a global seed using
            # current generator
            np.random.seed(_autopara_per_item_info[at_i]["rng"].integers(2 ** 32))

        traj = _atom_opt_hopping(atom=at, calculator=calculator, Ediff0=Ediff0, T0=T0, minima_threshold=minima_threshold,
                                 mdmin=mdmin, fmax=fmax, timestep=timestep, totalsteps=totalsteps,
                                 skip_failures=skip_failures, update_config_type=update_config_type, results_prefix=results_prefix,
                                 workdir=workdir, **opt_kwargs)
        all_trajs.append(traj)

    return all_trajs


# run that operation on ConfigSet, for multiprocessing
def minimahopping(*args, **kwargs):
    default_autopara_info = {"num_inputs_per_python_subprocess": 10}

    return autoparallelize(_run_autopara_wrappable, *args,
                           default_autopara_info=default_autopara_info, **kwargs)
autoparallelize_docstring(minimahopping, _run_autopara_wrappable, "Atoms")
