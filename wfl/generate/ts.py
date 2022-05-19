from tempfile import NamedTemporaryFile

import ase.io

import wfl.utils.misc
from wfl.generate import optimize
from wfl.utils.parallel import construct_calculator_picklesafe

try:
    from sella import Sella, IRC
except ModuleNotFoundError:
    Sella = None
    IRC = None


# noinspection PyProtectedMember
def calc_ts(atoms, calculator, fmax=1.0e-3, steps=200, traj_step_interval=1, traj_equispaced_n=None, verbose=False):
    """Runs TS calculation

    Notes
    -----
    - Pressure and cell movement is not supported yet
    - Constraints are not implemented, Sella needs special treatment of them,
      see: https://github.com/zadorlab/sella/wiki/Constraints
    - Keeping the symmetry is not supported by the Sella optimiser

    Parameters
    ----------
    atoms: list(Atoms)
        input configs
    calculator: Calculator / (initializer, args, kwargs)
        ASE calculator or routine to call to create calculator
    fmax: float, default 1e-3
        force convergence tolerance
    steps: int, default 200
        max number of steps
    traj_step_interval: int, default 1
        if present, interval between trajectory snapshots
    traj_equispaced_n: int, default None
        if present, number of configurations to save from trajectory,
        trying to be equispaced in Cartesian path length
    verbose: bool, default False
        optimisation logs are not printed unless this is True

    Returns
    -------
        list(Atoms) trajectories
    """
    if not Sella or not IRC:
        raise RuntimeError('Need Sella, IRC from sella module')

    if verbose:
        logfile = '-'
    else:
        logfile = None

    calculator = construct_calculator_picklesafe(calculator)

    all_trajs = []

    for at in wfl.utils.misc.atoms_to_list(atoms):

        at.calc = calculator
        at.constraints = None
        at.info['config_type'] = 'minim_traj'

        with NamedTemporaryFile(prefix="sella_", suffix="_.traj") as trajfile:
            opt = Sella(at, trajectory=trajfile.name, logfile=logfile)
            opt.run(fmax=fmax, steps=steps)

            traj = ase.io.read(trajfile.name, ":")
            if traj_step_interval is not None and traj_step_interval > 0:
                # enforce having the last frame in it
                traj = traj[:-1:traj_step_interval] + [traj[-1]]

            if opt.converged():
                traj[-1].info['config_type'] = 'ts_last_converged'
            else:
                traj[-1].info['config_type'] = 'ts_last_unconverged'

            traj = optimize._resample_traj(traj, traj_equispaced_n)

            all_trajs.append(traj)

    return all_trajs
