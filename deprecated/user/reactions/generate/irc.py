import ase.io
import numpy as np

try:
    from sella import IRC
except ModuleNotFoundError:
    IRC = None
from tempfile import NamedTemporaryFile

import wfl.utils.misc
from wfl.generate import optimize
from wfl.utils.parallel import construct_calculator_picklesafe


def calc_irc(atoms, calculator, fmax=0.1, steps=200, traj_step_interval=1, traj_equispaced_n=None, irc_kwargs=None,
             verbose=False):
    """Calculate IRC trajectory from TS geometries

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
    irc_kwargs: dict
        kwargs for IRC optimiser, default: {dx=0.1, eta=1e-4, gamma=0.4}
    verbose: bool, default False
        verbose output
        optimisation logs are not printed unless this is True

    Returns
    -------
    list(Atoms) trajectories

    """
    if not IRC:
        raise RuntimeError('need IRC from sella module')

    if irc_kwargs is None:
        irc_kwargs = dict(dx=0.1, eta=1e-4, gamma=0.4)

    if verbose:
        irc_kwargs["logfile"] = '-'
    else:
        irc_kwargs["logfile"] = None

    calculator = construct_calculator_picklesafe(calculator)

    all_trajs = []

    for at in wfl.utils.misc.atoms_to_list(atoms):

        at.calc = calculator
        at.info['config_type'] = 'irc_traj'

        with NamedTemporaryFile(prefix="sella_", suffix="_.traj") as trajfile:
            opt = IRC(at, trajectory=trajfile.name, **irc_kwargs)

            try:
                opt.run(fmax=fmax, steps=steps, direction='forward')
            except RuntimeError:
                print("Failed IRC forwards, Continuing anyways.")

            try:
                opt.run(fmax=fmax, steps=steps, direction='reverse')
            except RuntimeError:
                print("Failed IRC backwards, Continuing anyways.")

            traj = ase.io.read(trajfile.name, ":")

            # order the traj as reaction coordinate, the forward bit needs to be reversed
            energies = np.array([at.info["energy"] for at in traj])

            try:
                idx = np.argwhere(np.diff(energies) > 0.)[0][0]
                traj = traj[idx:][::-1] + traj[:idx]

                if traj_step_interval is not None and traj_step_interval > 0:
                    # enforce first, TS and last as well
                    traj = traj[:idx - 1:traj_step_interval] + traj[idx:-1:traj_step_interval] + [traj[-1]]
            except IndexError:
                print("failed to find index in energy array:", energies)

            if opt.converged():
                traj[-1].info['config_type'] = 'irc_last_converged'
                traj[0].info['config_type'] = 'irc_last_converged'
            else:
                traj[-1].info['config_type'] = 'irc_last_unconverged'
                traj[0].info['config_type'] = 'irc_last_unconverged'

            # noinspection PyProtectedMember
            traj = optimize._resample_traj(traj, traj_equispaced_n)

            all_trajs.append(traj)

    return all_trajs
