import sys

import numpy as np
from ase.optimize import FIRE
from ase.mep.dyneb import DyNEB
from ase.atoms import Atoms

from wfl.autoparallelize import autoparallelize, autoparallelize_docstring
from wfl.utils.save_calc_results import at_copy_save_calc_results
from wfl.utils.misc import atoms_to_list
from wfl.utils.parallel import construct_calculator_picklesafe
from .utils import save_config_type


def _run_autopara_wrappable(list_of_images, calculator, fmax=5e-2, steps=1000,
           traj_step_interval=1, traj_subselect=None, skip_failures=True, attach_kwargs=None, attach_interval=None,
           results_prefix='last_op__neb_', verbose=False, logfile=None, update_config_type="append", abort_check=None,
           **neb_kwargs):
    """runs a structure optimization. By default calculator properties will be stored
    in keys prefixed with "last_op__neb_", which may be overwritten by next operation.

    Parameters
    ----------
    list_of_images: list(iterable(Atoms))
        input list of images, with each is an itereable of Atoms
        (i.e. [images1, images2, images3])
    calculator: Calculator / (initializer, args, kwargs)
        ASE calculator or initializer and arguments to call to create calculator
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
    attach_kwargs : function, default None
        function to be attached to optimizer
    attach_interval : int, default None
        interval on which function is executed during optimization 
    results_prefix: str, default "last_op__neb_"
        prefix to info/arrays keys where calculator properties will be stored
        Will overwrite any other properties that start with same "<str>__", so that by
        default only last op's properties will be stored.
    verbose: bool, default False
        verbose output
        optimisation logs are not printed unless this is True
    update_config_type: ["append" | "overwrite" | False], default "append"
        whether/how to add at.info['neb_config_type'] to at.info['config_type']
    abort_check : default None, 
        wfl.generate.md.abort_base.AbortBase - derived class that
        checks the NEB snapshots and aborts the simulation on some condition.
    neb_kwargs
        keyword arguments for DyNEB and FIRE 

    Returns
    -------
        list(Atoms) trajectories
    """
    logfile = neb_kwargs.get("logfile", None)
    if attach_kwargs is not None:
        attach_constructor = attach_kwargs.pop("attach_function", None) 

    if logfile is None and verbose:
        logfile = "-"

    calculator = construct_calculator_picklesafe(calculator)

    all_trajs = []

    for images in list_of_images:
        assert all([isinstance(at, Atoms) for at in images]), "Got images that is not an iterable(Atoms)"

        orig_constraints = [at.constraints for at in images]
        for at_i, at in enumerate(images):
            at.calc = calculator
            at.info['neb_image_i'] = at_i

        neb = DyNEB(list(images), **neb_kwargs)
        opt = FIRE(neb, logfile=logfile)

        traj = []
        def process_step():

            cur_images = []
            for at, constraints in zip(images, orig_constraints):
                new_config = at_copy_save_calc_results(at, prefix=results_prefix)
                new_config.set_constraint(constraints)
                new_config.info['neb_iter_i'] = opt.get_number_of_steps()
                cur_images.append(new_config)

            traj.append(cur_images)
            
            if abort_check is not None:
                if abort_check.stop(opt):
                    raise RuntimeError(f"NEB was stopped by the NEB checker function {abort_check.__class__.__name__}")
        opt.attach(process_step, interval=traj_step_interval)

        if attach_kwargs is not None:
            attach_kwargs["neb"] = neb
            opt.attach(attach_constructor, interval=attach_interval, **attach_kwargs)

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

        for intermed_images in traj[1:-1]:
            for at in intermed_images:
                at.info['neb_config_type'] = 'neb_intermediate'

        for at in traj[-1]:
            at.info['neb_config_type'] = f'neb_last_{final_status}'
            at.info['neb_n_steps'] = opt.get_number_of_steps()

        for all_images in traj:
            for at in all_images:
                save_config_type(at, update_config_type, at.info['neb_config_type'])

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
        return traj[-1]
    elif subselect == "last_converged":
        return traj[-1] if traj[-1][0].info['neb_config_type'] == 'neb_last_converged' else None

    raise RuntimeError(f'Subselecting confgs from trajectory with rule '
                       f'"subselect={subselect}" is not yet implemented')
