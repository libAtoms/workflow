import os
import sys

import ase.units
import numpy as np
import spglib
from ase.constraints import ExpCellFilter
from ase.optimize.precon import PreconLBFGS

from wfl.autoparallelize import autoparallelize
from wfl.utils.at_copy_save_results import at_copy_save_results
from wfl.utils.misc import atoms_to_list
from wfl.utils.parallel import construct_calculator_picklesafe
from wfl.utils.pressure import sample_pressure
from .utils import config_type_append

orig_log = PreconLBFGS.log


def new_log(self, forces=None):
    if 'buildcell_config_i' in self.atoms.info:
        if self.logfile is not None:
            self.logfile.write(str(self.atoms.info['buildcell_config_i']) + ' ')
    orig_log(self, forces)

    try:
        self.logfile.flush()
    except:
        pass


PreconLBFGS.log = new_log


# run that operates on ConfigSet, for multiprocessing
def run(inputs, outputs, calculator, fmax=1.0e-3, smax=None, steps=1000, pressure=None,
        keep_symmetry=True, traj_step_interval=1, traj_subselect=None, skip_failures=True,
        results_prefix='minim_', chunksize=10, verbose=False, update_config_type=True, **opt_kwargs):
    # Normally each thread needs to call np.random.seed so that it will generate a different
    # set of random numbers.  This env var overrides that to produce deterministic output,
    # for purposes like testing
    if 'WFL_DETERMINISTIC_HACK' in os.environ:
        initializer = None
    else:
        initializer = np.random.seed
    return autoparallelize(iterable=inputs, configset_out=outputs, op=run_op, chunksize=chunksize,
                         calculator=calculator, fmax=fmax, smax=smax, steps=steps,
                         pressure=pressure, keep_symmetry=keep_symmetry, traj_step_interval=traj_step_interval,
                         traj_subselect=traj_subselect, skip_failures=skip_failures, results_prefix=results_prefix,
                         verbose=verbose, update_config_type=update_config_type,
                         initializer=initializer, hash_ignore=['initializer'], **opt_kwargs)


def run_op(atoms, calculator, fmax=1.0e-3, smax=None, steps=1000, pressure=None,
           keep_symmetry=True, traj_step_interval=1, traj_subselect=None, skip_failures=True,
           results_prefix='minim_', verbose=False, update_config_type=True, **opt_kwargs):
    """runs a minimization

    Parameters
    ----------
    atoms: list(Atoms)
        input configs
    calculator: Calculator / (initializer, args, kwargs)
        ASE calculator or routine to call to create calculator
    fmax: float, default 1e-3
        force convergence tolerance
    smax: float, default None
        stress convergence tolerance, default from fmax
    steps: int, default 1000
        max number of steps
    pressure: None / float / tuple
        applied pressure distribution (GPa), as parsed by wfl.utils.pressure.sample_pressure()
    keep_symmetry: bool, default True
        constrain symmetry to maintain initial
    traj_step_interval: int, default 1
        if present, interval between trajectory snapshots
    traj_subselect: "last_converged", default None
        rule for sub-selecting configs from the full trajectory.
        Currently implemented: "last_converged", which takes the last config, if converged.
    skip_failures: bool, default True
        just skip minimizations that raise an exception
    verbose: bool, default False
        verbose output
        optimisation logs are not printed unless this is True
    update_config_type: bool, default True
        append at.info['minim_config_type'] at.info['config_type']
    opt_kwargs
        keyword arguments for PreconLBFGS

    Returns
    -------
        list(Atoms) trajectories
    """

    opt_kwargs_to_use = dict(logfile=None, master=True)
    opt_kwargs_to_use.update(opt_kwargs)

    if opt_kwargs_to_use.get('logfile') is None and verbose:
        opt_kwargs_to_use['logfile'] = '-'

    calculator = construct_calculator_picklesafe(calculator)

    if smax is None:
        smax = fmax

    if keep_symmetry:
        # noinspection PyUnresolvedReferences,PyUnresolvedReferences
        from ase.spacegroup.symmetrize import FixSymmetry

    all_trajs = []

    for at in atoms_to_list(atoms):
        if keep_symmetry:
            sym = FixSymmetry(at)
            at.set_constraint(sym)
            dataset = spglib.get_symmetry_dataset((at.cell, at.get_scaled_positions(), at.numbers), 0.01)
            if 'buildcell_config_i' in at.info:
                print(at.info['buildcell_config_i'], end=' ')
            print('initial symmetry group number {}, international (Hermann-Mauguin) {} Hall {} prec {}'.format(
                dataset['number'], dataset['international'], dataset['hall'], 0.01))

        at.calc = calculator
        if pressure is not None:
            p = sample_pressure(pressure, at)
            at.info['minim_pressure_GPa'] = p
            p *= ase.units.GPa
            wrapped_at = ExpCellFilter(at, scalar_pressure=p)
        else:
            wrapped_at = at

        opt = PreconLBFGS(wrapped_at, **opt_kwargs_to_use)

        # default status, will be overwritten for first and last configs in traj
        at.info['minim_config_type'] = 'minim_mid'
        traj = []

        def process_step():
            if 'RSS_min_vol_per_atom' in at.info and at.get_volume() / len(at) < at.info['RSS_min_vol_per_atom']:
                raise RuntimeError('Got volume per atom {} under minimum {}'.format(at.get_volume() / len(at),
                                                                                    at.info['RSS_min_vol_per_atom']))

            if len(traj) > 0 and traj[-1] == at:
                # Some minimization algorithms sometimes seem to repeat, perhaps
                # only in weird circumstances, e.g. bad gradients near breakdown.
                # Do not store those duplicate configs.
                return

            traj.append(at_copy_save_results(at, results_prefix=results_prefix))

        opt.attach(process_step, interval=traj_step_interval)

        # preliminary value
        final_status = 'unconverged'

        try:
            opt.run(fmax=fmax, smax=smax, steps=steps)
        except Exception as exc:
            # label actual failed minims
            # when this happens, the atomic config somehow ends up with a 6-vector stress, which can't be
            # read by xyz reader.
            # that should probably never happen
            final_status = 'exception'
            if skip_failures:
                sys.stderr.write(f'Minimization failed with exception \'{exc}\'\n')
                sys.stderr.flush()
            else:
                raise

        if len(traj) == 0 or traj[-1] != at:
            traj.append(at_copy_save_results(at, results_prefix=results_prefix))

        # set for first config, to be overwritten if it's also last config
        traj[0].info['minim_config_type'] = 'minim_initial'

        if opt.converged():
            final_status = 'converged'

        traj[-1].info['minim_config_type'] = f'minim_last_{final_status}'
        traj[-1].info['minim_n_steps'] = opt.get_number_of_steps()

        if keep_symmetry:
            # should we check that initial is subgroup of final, i.e. no symmetry was lost?
            dataset = spglib.get_symmetry_dataset((traj[-1].cell, traj[-1].get_scaled_positions(), traj[-1].numbers), 0.01)
            if 'buildcell_config_i' in at.info:
                print(at.info['buildcell_config_i'], end=' ')
            if dataset is None:
                print('final symmetry group number None')
            else:
                print('final symmetry group number {}, international (Hermann-Mauguin) {} Hall {} prec {}'.format(
                    dataset['number'], dataset['international'], dataset['hall'], 0.01))


        if update_config_type:
            # save config_type
            for at0 in traj:
                config_type_append(at0, at0.info['minim_config_type'])

        # Note that if resampling doesn't include original last config, later
        # steps won't be able to identify those configs as the (perhaps unconverged) minima.
        # Perhaps status should be set after resampling?
        traj = subselect_from_traj(traj, subselect=traj_subselect)

        all_trajs.append(traj)

    return all_trajs


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
        None: full trajectory is returned
        int: (not implemented) how many samples to take from the trajectory.
        str:
            - "last_converged": returns [last_config], if converged or None if not.

    """
    if subselect is None:
        return traj

    elif subselect == "last_converged":
        converged_configs = [at for at in traj if at.info["minim_config_type"] == "minim_last_converged"]
        if len(converged_configs) == 0:
            return None
        else:
            return converged_configs

    raise RuntimeError(f'Subselecting confgs from trajectory with rule '
                       f'"subselect={subselect}" is not yet implemented')
