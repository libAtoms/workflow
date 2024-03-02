import sys

import numpy as np
from ase.optimize import FIRE
from ase.mep.dyneb import DyNEB

from wfl.autoparallelize import autoparallelize, autoparallelize_docstring
from wfl.utils.at_copy_save_results import at_copy_save_results
from wfl.utils.misc import atoms_to_list
from wfl.utils.parallel import construct_calculator_picklesafe
from .utils import config_type_append


def _run_autopara_wrappable(list_of_images, calculator, fmax=5e-2, steps=1000,
           traj_step_interval=1, traj_subselect=None, skip_failures=True,
           results_prefix='neb_', verbose=False, logfile=None, update_config_type=True,
           autopara_rng_seed=None, autopara_per_item_info=None,
           **neb_kwargs):
    """runs a structure optimization

    Parameters
    ----------
    list_of_images: list(images)
        input configs. images are list(Atoms)
        (i.e. [images1, images2, images3])
    calculator: Calculator / (initializer, args, kwargs)
        ASE calculator or routine to call to create calculator
    fmax: float, default 5e-2
        force convergence tolerance
    steps: int, default 1000
        max number of steps
    traj_step_interval: int, default 1
        if present, interval between trajectory snapshots
    traj_subselect: "last_converged", default None
        rule for sub-selecting configs from the full trajectory.
        Currently implemented: "last_converged", which takes the last config, if converged.
    skip_failures: bool, default True
        just skip optimizations that raise an exception
    verbose: bool, default False
        verbose output
        optimisation logs are not printed unless this is True
    update_config_type: bool, default True
        append at.info['optimize_config_type'] at.info['config_type']
    neb_kwargs
        keyword arguments for DyNEB and FIRE 
    autopara_rng_seed: int, default None
        global seed used to initialize rng so that each operation uses a different but
        deterministic local seed, use a random value if None

    Returns
    -------
        list(Atoms) trajectories
    """
    logfile = neb_kwargs.get("logfile", None)
   
    if logfile is None and verbose:
        logfile = "-"

    calculator = construct_calculator_picklesafe(calculator)

    all_trajs = []

    for images in list_of_images:
        for at_i, at in enumerate(atoms_to_list(images)):
            if autopara_per_item_info is not None:
                np.random.seed(autopara_per_item_info[at_i]["rng_seed"])
    
            # original constraints
            org_constraints = at.constraints
    
            at.calc = calculator

    neb = DyNEB(images, **neb_kwargs)
    opt = FIRE(neb, logfile=logfile)

    traj = []
    def process_step():

        subtraj = []
        for at in images:
            new_config = at_copy_save_results(at, results_prefix=results_prefix)
            new_config.set_constraint(org_constraints)
            subtraj.append(new_config)

        traj.append(subtraj)

    opt.attach(process_step, interval=traj_step_interval)

    # preliminary value
    final_status = 'unconverged'

    try:
        opt.run(fmax=fmax, steps=steps)
    except Exception as exc:
        # label actual failed optimizations
        # when this happens, the atomic config somehow ends up with a 6-vector stress, which can't be
        # read by xyz reader.
        # that should probably never happen
        final_status = 'exception'
        if skip_failures:
            sys.stderr.write(f'Nudged elastic band calculation failed with exception \'{exc}\'\n')
            sys.stderr.flush()
        else:
            raise

    # set for first config, to be overwritten if it's also last config

    for at in traj[0]:
        at.info['neb_config_type'] = 'neb_initial'

    if opt.converged():
        final_status = 'converged'

    for subtraj in traj[1:-1]:
        for at in subtraj:
            at.info['neb_config_type'] = f'neb_intermediate'

    for at in traj[-1]:
        at.info['neb_config_type'] = f'neb_last_{final_status}'
        at.info['neb_n_steps'] = opt.get_number_of_steps()


    if update_config_type:
        # save config_type
        for subtraj in traj:
            for at0 in subtraj:
                config_type_append(at0, at0.info['neb_config_type'])

    # Note that if resampling doesn't include original last config, later
    # steps won't be able to identify those configs as the (perhaps unconverged) minima.
    # Perhaps status should be set after resampling?
    traj = subselect_from_traj(traj, subselect=traj_subselect)

    all_trajs.append(traj)
    
    return all_trajs


def NEB(*args, **kwargs):
    default_autopara_info = {"num_inputs_per_python_subprocess": 10}

    return autoparallelize(_run_autopara_wrappable, *args,
                           default_autopara_info=default_autopara_info, **kwargs)
autoparallelize_docstring(NEB, _run_autopara_wrappable, "Atoms")


def subselect_from_traj(traj, subselect=None):
    """Sub-selects configurations from trajectory.

    Parameters
    ----------
    subselect: int or string, default None

        - None: full trajectory is returned
        - int: (not implemented) how many samples to take from the trajectory.
        - str: specific method

          - "last_converged": returns [last_config] if converged, or None if not.

    """
    if subselect is None:
        return traj
    elif subselect == "last":
        return [at for subtraj in traj for at in subtraj if at.info["neb_config_type"] == "neb_last_unconverged"]
    elif subselect == "last_converged":
        converged_configs = [at for subtraj in traj for at in subtraj if at.info["neb_config_type"] == "neb_last_converged"]
        if len(converged_configs) == 0:
            return None
        else:
            return converged_configs

    raise RuntimeError(f'Subselecting confgs from trajectory with rule '
                       f'"subselect={subselect}" is not yet implemented')
