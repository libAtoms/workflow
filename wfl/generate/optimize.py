import sys

import ase.units
import numpy as np
import spglib
from ase.filters import FrechetCellFilter
from ase.optimize.precon import PreconLBFGS

from wfl.autoparallelize import autoparallelize, autoparallelize_docstring
from wfl.utils.at_copy_save_results import at_copy_save_results
from wfl.utils.misc import atoms_to_list
from wfl.utils.parallel import construct_calculator_picklesafe
from wfl.utils.pressure import sample_pressure
from .utils import config_type_append

orig_log = PreconLBFGS.log


def _new_log(self, forces=None):
    if 'buildcell_config_i' in self.atoms.info:
        if self.logfile is not None:
            self.logfile.write(str(self.atoms.info['buildcell_config_i']) + ' ')
    orig_log(self, forces)

    try:
        self.logfile.flush()
    except:
        pass


PreconLBFGS.log = _new_log


def _run_autopara_wrappable(atoms, calculator, fmax=1.0e-3, smax=None, steps=1000, pressure=None,
           stress_mask=None, keep_symmetry=True, traj_step_interval=1, traj_subselect=None,
           skip_failures=True, results_prefix='optimize_', verbose=False, update_config_type=True,
           rng=None, _autopara_per_item_info=None,
           **opt_kwargs):
    """runs a structure optimization

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
    stress_mask: None / list(bool)
        mask for stress components to pass to variable-cell filter
    keep_symmetry: bool, default True
        constrain symmetry to maintain initial
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
        keyword arguments for PreconLBFGS
    rng: numpy.random.Generator, default None
        random number generator to use (needed for pressure sampling, initial temperature, or Langevin dynamics)
    _autopara_per_item_info: dict
        INTERNALLY used by autoparallelization framework to make runs reproducible (see
        wfl.autoparallelize.autoparallelize() docs)

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
        try:
            from ase.constraints import FixSymmetry
        except ImportError:
            # fall back to previous import location (pre MR 3288)
            from ase.spacegroup.symmetrize import FixSymmetry

    all_trajs = []

    for at_i, at in enumerate(atoms_to_list(atoms)):
        # get rng from autopara_per_item info if available ("rng" arg that was passed in was
        # already used by autoparallelization framework to set "rng" key in per-item dict)
        rng = _autopara_per_item_info[at_i].get("rng")

        # original constraints
        org_constraints = at.constraints

        if keep_symmetry:
            sym = FixSymmetry(at, adjust_cell=False)
            # Append rather than overwrite constraints
            at.set_constraint([*at.constraints, sym])

            dataset = spglib.get_symmetry_dataset((at.cell, at.get_scaled_positions(), at.numbers), 0.01)
            if 'buildcell_config_i' in at.info:
                print(at.info['buildcell_config_i'], end=' ')
            print('initial symmetry group number {}, international (Hermann-Mauguin) {} Hall {} prec {}'.format(
                dataset['number'], dataset['international'], dataset['hall'], 0.01))

        at.calc = calculator
        if pressure is not None:
            p = sample_pressure(pressure, at, rng=rng)
            at.info['optimize_pressure_GPa'] = p
            p *= ase.units.GPa
            wrapped_at = FrechetCellFilter(at, scalar_pressure=p, mask=stress_mask)
        else:
            wrapped_at = at

        opt = PreconLBFGS(wrapped_at, **opt_kwargs_to_use)

        # default status, will be overwritten for first and last configs in traj
        at.info['optimize_config_type'] = 'optimize_mid'
        traj = []

        def process_step():
            if 'RSS_min_vol_per_atom' in at.info and at.get_volume() / len(at) < at.info['RSS_min_vol_per_atom']:
                raise RuntimeError('Got volume per atom {} under minimum {}'.format(at.get_volume() / len(at),
                                                                                    at.info['RSS_min_vol_per_atom']))

            if len(traj) > 0 and traj[-1] == at:
                # Some optimization algorithms sometimes seem to repeat, perhaps
                # only in weird circumstances, e.g. bad gradients near breakdown.
                # Do not store those duplicate configs.
                return

            new_config = at_copy_save_results(at, results_prefix=results_prefix)
            new_config.set_constraint(org_constraints)
            traj.append(new_config)

        opt.attach(process_step, interval=traj_step_interval)

        # preliminary value
        final_status = 'unconverged'

        try:
            opt.run(fmax=fmax, smax=smax, steps=steps)
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

        if len(traj) == 0 or traj[-1] != at:
            new_config = at_copy_save_results(at, results_prefix=results_prefix)
            new_config.set_constraint(org_constraints)
            traj.append(new_config)

        # set for first config, to be overwritten if it's also last config
        traj[0].info['optimize_config_type'] = 'optimize_initial'

        if opt.converged():
            final_status = 'converged'

        traj[-1].info['optimize_config_type'] = f'optimize_last_{final_status}'
        traj[-1].info['optimize_n_steps'] = opt.get_number_of_steps()

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
                config_type_append(at0, at0.info['optimize_config_type'])

        # Note that if resampling doesn't include original last config, later
        # steps won't be able to identify those configs as the (perhaps unconverged) minima.
        # Perhaps status should be set after resampling?
        traj = subselect_from_traj(traj, subselect=traj_subselect)

        all_trajs.append(traj)

    return all_trajs


def optimize(*args, **kwargs):
    default_autopara_info = {"num_inputs_per_python_subprocess": 10}

    return autoparallelize(_run_autopara_wrappable, *args,
                           default_autopara_info=default_autopara_info, **kwargs)
autoparallelize_docstring(optimize, _run_autopara_wrappable, "Atoms")


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

          - "last": returns [last_config]
          - "last_converged": returns [last_config] if converged, or None if not.

    """
    if subselect is None:
        return traj
    elif subselect == "last":
        return traj[-1]
    elif subselect == "last_converged":
        return traj[-1] if (traj[-1].info["optimize_config_type"] == "optimize_last_converged") else None

    raise RuntimeError(f'Subselecting confgs from trajectory with rule '
                       f'"subselect={subselect}" is not yet implemented')
