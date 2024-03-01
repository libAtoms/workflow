import sys

import ase.units
import numpy as np
import spglib
from ase.constraints import ExpCellFilter
#from ase.optimize.precon import PreconFIRE
from ase.optimize import FIRE
from ase.neb import NEB
from ase.dyneb import DyNEB

from wfl.autoparallelize import autoparallelize, autoparallelize_docstring
from wfl.utils.at_copy_save_results import at_copy_save_results
from wfl.utils.misc import atoms_to_list
from wfl.utils.parallel import construct_calculator_picklesafe
from wfl.utils.pressure import sample_pressure
from .utils import config_type_append

orig_log = FIRE.log


def _new_log(self, forces=None):
    if 'buildcell_config_i' in self.atoms.info:
        if self.logfile is not None:
            self.logfile.write(str(self.atoms.info['buildcell_config_i']) + ' ')
    orig_log(self, forces)

    try:
        self.logfile.flush()
    except:
        pass


#PreconFIRE.log = _new_log


def _run_autopara_wrappable(images, calculator, fmax=1.0e-3, smax=None, steps=1000, pressure=None,
           keep_symmetry=True, traj_step_interval=1, traj_subselect=None, skip_failures=True,
           results_prefix='optimize_', verbose=False, update_config_type=True,
           autopara_rng_seed=None, autopara_per_item_info=None,
           **opt_kwargs):
    """runs a structure optimization

    Parameters
    ----------
    images: list(Atoms)
        input configs
    calculator: Calculator / (initializer, args, kwargs)
        ASE calculator or routine to call to create calculator
    fmax: float, default 1e-3
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
    opt_kwargs
        keyword arguments for PreconFIRE
    autopara_rng_seed: int, default None
        global seed used to initialize rng so that each operation uses a different but
        deterministic local seed, use a random value if None

    Returns
    -------
        list(Atoms) trajectories
    """
    opt_kwargs_to_use = dict(logfile=None, master=True)
    opt_kwargs_to_use.update(opt_kwargs)

    if opt_kwargs_to_use.get('logfile') is None and verbose:
        opt_kwargs_to_use['logfile'] = '-'

    calculator = construct_calculator_picklesafe(calculator)

    all_trajs = []

    for at_i, at in enumerate(atoms_to_list(images)):
        if autopara_per_item_info is not None:
            np.random.seed(autopara_per_item_info[at_i]["rng_seed"])

        # original constraints
        org_constraints = at.constraints

        at.calc = calculator

    neb = DyNEB(images, k=0.2, climb=True, allow_shared_calculator=True, scale_fmax = 1)
    opt = FIRE(neb, **opt_kwargs_to_use)

        # default status, will be overwritten for first and last configs in traj
#        at.info['optimize_config_type'] = 'optimize_mid'

    traj = []
    def process_step():

#            if len(traj) > 0 and traj[-1] == at:
#                # Some optimization algorithms sometimes seem to repeat, perhaps
#                # only in weird circumstances, e.g. bad gradients near breakdown.
#                # Do not store those duplicate configs.
#                return
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
            sys.stderr.write(f'Structure optimization failed with exception \'{exc}\'\n')
            sys.stderr.flush()
        else:
            raise

#    if len(traj) == 0 or traj[-1] != at:
#        new_config = at_copy_save_results(at, results_prefix=results_prefix)
#        new_config.set_constraint(org_constraints)
#        traj.append(new_config)

    # set for first config, to be overwritten if it's also last config
#    print("traj : ", traj)

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
    
    print("Hello!")
    return all_trajs


def NEB(*args, **kwargs):
    default_autopara_info = {"num_inputs_per_python_subprocess": 10}

    return autoparallelize(_run_autopara_wrappable, *args,
                           default_autopara_info=default_autopara_info, **kwargs)
#autoparallelize_docstring(NEB, _run_autopara_wrappable, "Atoms")


# Just a placeholder for now. Could perhaps include:
#    equispaced in energy
#    equispaced in Cartesian path length
#    equispaced in some other kind of distance (e.g. SOAP)
# also, should it also have max distance instead of number of samples?
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

    elif subselect == "last_converged":
        converged_configs = [at for subtraj in traj for at in subtraj if at.info["neb_config_type"] == "neb_last_converged"]
        if len(converged_configs) == 0:
            return None
        else:
            return converged_configs

    raise RuntimeError(f'Subselecting confgs from trajectory with rule '
                       f'"subselect={subselect}" is not yet implemented')
