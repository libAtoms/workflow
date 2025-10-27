import sys
import json

import numpy as np
from ase.md.nptberendsen import NPTBerendsen
from ase.md.nvtberendsen import NVTBerendsen
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution, Stationary
from ase.md.verlet import VelocityVerlet
from ase.md.langevin import Langevin
from ase.md.logger import MDLogger
from ase.units import fs

# only in 3.27.0b1 as of 2025 Oct 23 or so
try:
    from ase.md.langevinbaoab import LangevinBAOAB
except ImportError:
    LangevinBAOAB = None

from wfl.autoparallelize import autoparallelize, autoparallelize_docstring
from wfl.utils.save_calc_results import at_copy_save_calc_results
from wfl.utils.misc import atoms_to_list
from wfl.utils.parallel import construct_calculator_picklesafe
from ..utils import save_config_type
from .utils import _get_temperature, _get_pressure


def _sample_autopara_wrappable(atoms, calculator, steps, dt, integrator="Berendsen", temperature=None, temperature_tau=None,
              pressure=None, hydrostatic=True, pressure_tau=None, pressure_mass_factor=1, compressibility_au=None, compressibility_fd_displ=0.01,
              traj_step_interval=1, skip_failures=True, results_prefix='last_op__md_', verbose=False, update_config_type="append",
              traj_select_during_func=lambda at: True, traj_select_after_func=None, abort_check=None,
              logger_interval=0, logger_kwargs=None, rng=None, _autopara_per_item_info=None):
    """runs an MD trajectory with aggresive, not necessarily physical, integrators for
    sampling configs. By default calculator properties for each frame stored in
    keys prefixed with "last_op__md_", which may be overwritten by next operation.
    Most keyword args (all except `calculator`, `steps`, `dt`, and `logger_*`) can be
    overridden per-config by a json-encoded dict in each `atoms.info["WFL_MD_KWARGS"]`

    Parameters
    ----------
    atoms: list(Atoms)
        input configs
    calculator: Calculator / (initializer, args, kwargs)
        ASE calculator or routine to call to create calculator
    dt: float
        time step (fs)
    integrator: "Berendsen" or "Langevin" or "LangevinBAOAB", default "Berendsen"
        MD time integrator. Berendsen and Langevin fall back to VelocityVerlet (NVE) if temperature_tau is None,
        while LangevinBAOAB will do NPH in that case
    steps: int
        number of steps
    temperature: float or (float, float, [int]]), or list of dicts  default None
        temperature control (Kelvin)
        - float: constant T
        - tuple/list of float, float, [int=10]: T_init, T_final, and optional number of stages for ramp
        - [ {'T_i': float, 'T_f' : float, 'traj_frac' : flot, 'n_stages': int=10}, ... ] list of stages, each one a ramp, with
          duration defined as fraction of total number of steps
    temperature_tau: float, default None
        Time scale for thermostat (fs). Directly used for Berendsen integrator, or as 1/friction for Langevin integrator.
    pressure: None / float / tuple
        Applied pressure distribution (GPa) as parsed by wfl.utils.pressure.sample_pressure().
        Enables Berendsen constant P volume rescaling.
    hydrostatic: bool, default True
        Allow only hydrostatic strain for variable cell (pressure not None)
    pressure_tau: float, default None
        time scale for Berendsen constant P volume rescaling (fs)
        ignored if pressure is None, defaults to 3*temperature_tau
    pressure_mass_factor: float, default 1
        factor to multiply by default pressure mass heuristic
    compressibility_au: float, default None
        compressibility, if available, for NPTBerendsen
    compressibility_fd_displ: float, default 0.01
        finite difference in strain to use when computing compressibility for NPTBerendsen
    traj_step_interval: int, default 1
        interval between trajectory snapshots
    skip_failures: bool, default True
        just skip minimizations that raise an exception
    results_prefix: str, default "last_op__md_"
        prefix to info/arrays keys where calculator properties will be stored
        Will overwrite any other properties that start with same "<str>__", so that by
        default only last op's properties will be stored.
    verbose: bool, default False
        verbose output (MD logging is handled by setting logger_interval > 0)
    update_config_type: ["append" | "overwrite" | False], default "append"
        whether/how to add 'MD' to at.info['config_type']
    traj_select_during_func: func(Atoms), default func(Atoms) -> bool=True
        Function to sub-select configs from the first trajectory.
        Used during MD loop with one config at a time, returning True/False
    traj_select_after_func: func(list(Atoms)), default None
        Function to sub-select configs from the first trajectory.
        Used at end of MD loop with entire trajectory as list, returns subset
    abort_check: default None,
        wfl.generate.md.abort_base.AbortBase - derived class that
        checks the MD snapshots and aborts the simulation on some condition.
    rng: numpy.random.Generator, default None
        random number generator to use (needed for pressure sampling, initial temperature, or Langevin dynamics)
    logger_interval: int, default 0
        Enable logging at this interval if > 0
    logger_kwargs: dict, default None
        kwargs to MDLogger to attach to each MD run, including "logfile" as string to which
        config number will be appended. If logfile is "-", stdout will be used, and config number
        will be prepended to each outout line. User defined ase.md.MDLogger derived class can be provided with "logger" as key.
    _autopara_per_item_info: dict
        INTERNALLY used by autoparallelization framework to make runs reproducible (see
        wfl.autoparallelize.autoparallelize() docs)

    Returns
    -------
        list(Atoms) trajectories
    """

    return _sample_autopara_wrappable_kwargs(atoms, calculator, steps, dt,
            integrator=integrator,
            temperature=temperature,
            temperature_tau=temperature_tau,
            pressure=pressure,
            hydrostatic=hydrostatic,
            pressure_tau=pressure_tau,
            pressure_mass_factor=pressure_mass_factor,
            compressibility_au=compressibility_au,
            compressibility_fd_displ=compressibility_fd_displ,
            traj_step_interval=traj_step_interval,
            skip_failures=skip_failures,
            results_prefix=results_prefix,
            verbose=verbose,
            update_config_type=update_config_type,
            traj_select_during_func=traj_select_during_func,
            traj_select_after_func=traj_select_after_func,
            abort_check=abort_check,
            logger_interval=logger_interval,
            logger_kwargs=logger_kwargs,
            rng=rng,
            _autopara_per_item_info=_autopara_per_item_info)


def _sample_autopara_wrappable_single(at, at_i, calculator, steps, dt, logger_interval, logger_constructor, logger_logfile, logger_kwargs, integrator="Berendsen", temperature=None, temperature_tau=None,
              pressure=None, hydrostatic=True, pressure_tau=None, pressure_mass_factor=1, compressibility_au=None, compressibility_fd_displ=0.01,
              traj_step_interval=1, skip_failures=True, results_prefix='last_op__md_', verbose=False, update_config_type="append",
              traj_select_during_func=lambda at: True, traj_select_after_func=None, abort_check=None,
              rng=None, _autopara_per_item_info=None):
    # get rng from autopara_per_item info if available ("rng" arg that was passed in was
    # already used by autoparallelization framework to set "rng" key in per-item dict)
    rng = _autopara_per_item_info[at_i].get("rng")
    item_i = _autopara_per_item_info[at_i].get("item_i")

    at.calc = calculator
    temperature_use = _get_temperature(temperature, at, steps)
    pressure_use, compressibility_au_use = _get_pressure(pressure, compressibility_au, compressibility_fd_displ, at, rng)

    if temperature_use is not None:
        # set initial temperature
        assert rng is not None
        MaxwellBoltzmannDistribution(at, temperature_K=temperature_use[0]['T_i'], force_temp=True, communicator=None, rng=rng)
        Stationary(at, preserve_temperature=True)

    stage_kwargs = {'timestep': dt * fs}

    if integrator == "LangevinBAOAB":
        md_constructor = LangevinBAOAB
    else:
        if temperature_tau is None:
            md_constructor = VelocityVerlet
        else:
            if integrator == 'Langevin':
                md_constructor = Langevin
            elif integrator == 'Berendsen':
                if pressure_use is None:
                    md_constructor = NVTBerendsen
                else:
                    md_constructor = NPTBerendsen
            else:
                raise ValueError(f'Unkown integrator {integrator}')

    # pressure arguments, relatively simple because there are no stages
    if pressure_use is not None:
        if integrator == 'LangevinBAOAB':
            stage_kwargs['externalstress'] = pressure_use
            stage_kwargs['P_tau'] = pressure_tau * fs if pressure_tau is not None else temperature_tau * fs * 3
            stage_kwargs['P_mass_factor'] = pressure_mass_factor
            stage_kwargs['hydrostatic'] = hydrostatic
        elif integrator == 'Berendsen':
            if temperature_tau is None:
                raise ValueError('integrator Berendsen got pressure but no temperature_tau')
            if not hydrostatic:
                raise ValueError('integrator Berendsen got hydrostatic False')
            stage_kwargs['pressure_au'] = pressure_use
            stage_kwargs['compressibility_au'] = compressibility_au_use
            stage_kwargs['taup'] = pressure_tau * fs if pressure_tau is not None else temperature_tau * fs * 3
        else:
            raise ValueError(f'Only LangevinBAOAB and Berendsen integrator support pressure, got {integrator}')

    # temperature args except actual temperature, which is set below with stages
    if temperature_tau is not None:
        if integrator == 'LangevinBAOAB':
            stage_kwargs['T_tau'] = temperature_tau * fs
            assert rng is not None
            stage_kwargs["rng"] = rng
        elif integrator == 'Berendsen':
            stage_kwargs['taut'] = temperature_tau * fs
        elif integrator == 'Langevin':
            stage_kwargs["friction"] = 1 / (temperature_tau * fs)
            assert rng is not None
            stage_kwargs["rng"] = rng
        else:
            assert False, f'Unknown integrator {integrator}'

    if temperature_tau is None:
        # relatively simple, one stage
        all_stage_kwargs = [stage_kwargs.copy()]
        all_run_kwargs = [{'steps': steps}]
    else:
        # set up temperature stages
        all_stage_kwargs = []
        all_run_kwargs = []

        for t_stage_i, t_stage in enumerate(temperature_use):
            stage_steps = t_stage['traj_frac'] * steps

            if t_stage['T_f'] == t_stage['T_i']:
                # constant T
                stage_kwargs['temperature_K'] = t_stage['T_i']
                all_stage_kwargs.append(stage_kwargs.copy())
                all_run_kwargs.append({'steps': int(np.round(stage_steps))})
            else:
                # ramp
                for T in np.linspace(t_stage['T_i'], t_stage['T_f'], t_stage['n_stages']):
                    stage_kwargs['temperature_K'] = T
                    all_stage_kwargs.append(stage_kwargs.copy())
                substage_steps = int(np.round(stage_steps / t_stage['n_stages']))
                all_run_kwargs.extend([{'steps': substage_steps}] * t_stage['n_stages'])

    traj = []
    cur_step = 1
    first_step_of_later_stage = False

    def process_step(interval):
        nonlocal cur_step, first_step_of_later_stage

        if not first_step_of_later_stage and cur_step % interval == 0:
            at.info['MD_time_fs'] = cur_step * dt
            at.info['MD_step'] = cur_step
            at.info["MD_current_temperature"] = at.get_temperature()
            at_save = at_copy_save_calc_results(at, prefix=results_prefix)

            if traj_select_during_func(at):
                traj.append(at_save)

            if abort_check is not None:
                if abort_check.stop(at):
                    raise RuntimeError(f"MD was stopped by the MD checker function {abort_check.__class__.__name__}")

        first_step_of_later_stage = False
        cur_step += 1

    for stage_i, (stage_kwargs, run_kwargs) in enumerate(zip(all_stage_kwargs, all_run_kwargs)):
        if verbose:
            print('run stage', stage_kwargs, run_kwargs)

        # avoid double counting of steps at end of each stage and beginning of next
        cur_step -= 1

        if temperature_tau is not None:
            at.info['MD_temperature_K'] = stage_kwargs['temperature_K']

        md = md_constructor(at, **stage_kwargs)

        md.attach(process_step, 1, traj_step_interval)
        if logger_interval > 0:
            if logger_logfile == "-":
                logger_kwargs["logfile"] = "-"
            else:
                logger_kwargs["logfile"] = f"{logger_logfile}.config_{item_i}"
            logger_kwargs["dyn"] = md
            logger_kwargs["atoms"] = at
            logger = logger_constructor(**logger_kwargs)
            if logger_kwargs["logfile"] == "-":
                # add prefix to each line
                logger.hdr = f"config {item_i} " + logger.hdr
                logger.fmt = f"config {item_i} " + logger.fmt
            md.attach(logger, logger_interval)

        if stage_i > 0:
            first_step_of_later_stage = True

        try:
            md.run(**run_kwargs)
        except Exception as exc:
            if skip_failures:
                sys.stderr.write(f'MD failed with exception \'{exc}\'\n')
                sys.stderr.flush()
                break
            else:
                raise

    if len(traj) == 0 or traj[-1] != at:
        if traj_select_during_func(at):
            at.info['MD_time_fs'] = cur_step * dt
            traj.append(at_copy_save_calc_results(at, prefix=results_prefix))

    if traj_select_after_func is not None:
        traj = traj_select_after_func(traj)

    for at in traj:
        save_config_type(at, update_config_type, 'MD')

    return traj


def _sample_autopara_wrappable_kwargs(atoms, calculator, steps, dt, **kwargs):
    if kwargs.get("integrator", "Berendsen") not in ["Berendsen", "Langevin", "LangevinBAOAB"]:
        raise ValueError(f"Unknown integrator {kwargs['integrator']}, not Berendsen, Langevin, or LangevinBAOAB")

    calculator = construct_calculator_picklesafe(calculator)

    logger_interval = kwargs.pop("logger_interval", 0)
    logger_kwargs = kwargs.pop("logger_kwargs", None)
    if logger_kwargs is None:
        logger_kwargs = {}
    else:
        logger_kwargs = logger_kwargs.copy()
    logger_constructor = None
    logger_logfile = None
    if logger_interval > 0:
        logger_constructor = logger_kwargs.pop("logger", MDLogger)
        logger_logfile = logger_kwargs.pop("logfile", "-")

    kwargs_at = kwargs.copy()
    kwargs_orig = {}

    all_trajs = []
    for at_i, at in enumerate(atoms_to_list(atoms)):
        # reset kwargs_at to orig values
        kwargs_at.update(kwargs_orig)
        # set overriding values
        for k, v in json.loads(at.info.get("WFL_MD_KWARGS", "{}")).items():
            if k not in kwargs_orig:
                kwargs_orig[k] = kwargs[k]
            kwargs_at[k] = v

        # do operation with overridden values
        traj = _sample_autopara_wrappable_single(at, at_i, calculator, steps, dt,
                logger_interval, logger_constructor, logger_logfile, logger_kwargs, **kwargs_at)

        all_trajs.append(traj)

    return all_trajs


def md(*args, **kwargs):
    default_autopara_info = {"num_inputs_per_python_subprocess": 10}

    return autoparallelize(_sample_autopara_wrappable, *args,
                           default_autopara_info=default_autopara_info, **kwargs)
autoparallelize_docstring(md, _sample_autopara_wrappable, "Atoms")
