import os
import shutil
import sys
import tempfile

import numpy as np
from ase.optimize.minimahopping import MinimaHopping
from ase.io.trajectory import Trajectory

from wfl.autoparallelize import autoparallelize, autoparallelize_docstring
from wfl.utils.misc import atoms_to_list
from wfl.utils.parallel import construct_calculator_picklesafe


# perform MinimaHopping on one ASE.atoms object
def atom_opt_hopping(atom, calculator, Ediff0, T0, minima_threshold, mdmin, fmax, timestep, totalsteps, skip_failures, **opt_kwargs):
    workdir = os.getcwd()
    rundir = tempfile.mkdtemp(dir=workdir, prefix='Opt_hopping_')
    os.chdir(rundir)
    atom.calc = calculator
    try:
        opt = MinimaHopping(atom, Ediff0=Ediff0, T0=T0, minima_threshold=minima_threshold, mdmin=mdmin, fmax=fmax, timestep=timestep, **opt_kwargs)
        opt(totalsteps = totalsteps)
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
        for hop_traj in Trajectory('minima.traj'):
            hop_traj.info['config_type']='hopping_traj'
            traj.append(hop_traj)
        os.chdir(workdir)
        shutil.rmtree(rundir)
        return traj

def run_autopara_wrappable(atoms, calculator, Ediff0=1, T0=1000, minima_threshold=0.5, mdmin=2, fmax=1, timestep=1, totalsteps=10, skip_failures=True, **opt_kwargs):
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
    Returns
    -------
        list(Atoms) trajectories
    """

    calculator = construct_calculator_picklesafe(calculator)
    all_trajs = []

    for at in atoms_to_list(atoms):
        traj = []
        traj = atom_opt_hopping(atom=at, calculator=calculator, Ediff0=Ediff0, T0=T0, minima_threshold=minima_threshold, mdmin=mdmin,
                                fmax=fmax, timestep=timestep, totalsteps=totalsteps, skip_failures=skip_failures, **opt_kwargs)
        if traj is not None:
            all_trajs.append(traj)

    return all_trajs

# run that operation on ConfigSet, for multiprocessing
def run(*args, **kwargs):
    # Normally each thread needs to call np.random.seed so that it will generate a different
    # set of random numbers.  This env var overrides that to produce deterministic output,
    # for purposes like testing
    if 'WFL_DETERMINISTIC_HACK' in os.environ:
        initializer = (None, [])
    else:
        initializer = (np.random.seed, [])
    def_autopara_info={"initializer":initializer, "num_inputs_per_python_subprocess":10,
            "hash_ignore":["initializer"]}

    return autoparallelize(run_autopara_wrappable, *args,
        def_autopara_info=def_autopara_info, **kwargs)
run.__doc__ = autoparallelize_docstring(run_autopara_wrappable.__doc__, "Atoms")

