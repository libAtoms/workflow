import os
import sys

import numpy as np
from ase.md.nptberendsen import NPTBerendsen
from ase.md.nvtberendsen import NVTBerendsen
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution, Stationary
from ase.md.verlet import VelocityVerlet
from ase.units import GPa, fs, kB

from wfl.autoparallelize import autoparallelize
from wfl.utils.at_copy_save_results import at_copy_save_results
from wfl.utils.misc import atoms_to_list
from wfl.utils.parallel import construct_calculator_picklesafe
from wfl.utils.pressure import sample_pressure
from .utils import config_type_append

from util import configs

bar = 1.0e-4 * GPa


# run that operates on ConfigSet, for multiprocessing
def sample(inputs, outputs, calculator, steps, dt,
           temperature=None, temperature_tau=None, pressure=None, pressure_tau=None,
           compressibility_fd_displ=0.01,
           traj_step_interval=1, skip_failures=True, results_prefix='md_',
           chunksize=1, verbose=False, npool=None,update_config_type=True,
           selector_function=None, remote_info=None):
    # Normally each thread needs to call np.random.seed so that it will generate a different
    # set of random numbers.  This env var overrides that to produce deterministic output,
    # for purposes like testing
    if 'WFL_DETERMINISTIC_HACK' in os.environ:
        initializer = None
    else:
        initializer = np.random.seed
    return autoparallelize(iterable=inputs, outputspec=outputs, op=sample_autopara_wrappable, chunksize=chunksize,
                         calculator=calculator, steps=steps, dt=dt,
                         temperature=temperature, temperature_tau=temperature_tau,
                         pressure=pressure, pressure_tau=pressure_tau,
                         compressibility_fd_displ=compressibility_fd_displ,
                         traj_step_interval=traj_step_interval, skip_failures=skip_failures,
                         results_prefix=results_prefix, verbose=verbose, initializer=initializer, npool=npool, update_config_type=update_config_type, selector_function=selector_function, remote_info=remote_info)


def sample_autopara_wrappable(atoms, calculator, steps, dt, temperature=None, temperature_tau=None,
              pressure=None, pressure_tau=None, compressibility_fd_displ=0.01,
              traj_step_interval=1,  skip_failures=True, results_prefix='md_', verbose=False,
              selector_function=None, update_config_type=True):
    """runs an MD trajectory with aggresive, not necessarily physical, integrators for
    sampling configs

    Parameters
    ----------
    atoms: list(Atoms)
        input configs
    calculator: Calculator / (initializer, args, kwargs)
        ASE calculator or routine to call to create calculator
    dt: float
        time step (fs)
    steps: int
        number of steps
    temperature: float or (float, float, [int]]), or list of dicts  default None
        temperature control.  
        - float: constant T
        - tuple/list of float, float, [int=10]: T_init, T_final, and optional number of stages for ramp
        - [ {'T_i': float, 'T_f' : float, 'traj_frac' : flot, 'n_stages': int=10}, ... ] list of stages, each one a ramp, with duration
          defined as fraction of total number of steps
    temperature_tau: float, default None
        time scale that enables Berendsen constant T temperature rescaling (fs)
    pressure: None / float / tuple
        applied pressure distribution (GPa) as parsed by wfl.utils.pressure.sample_pressure()
        enabled Berendsen constant P volume rescaling
    pressure_tau: float, default None
        time scale for Berendsen constant P volume rescaling (fs)
        ignored if pressure is None, defaults to 3*temperature_tau
    compressibility_fd_displ: float, default 0.01
        finite difference in strain to use when computing compressibility for NPTBerendsen
    traj_step_interval: int, default 1
        interval between trajectory snapshots
    skip_failures: bool, default True
        just skip minimizations that raise an exception
    verbose: bool, default False
        verbose output
        MD logs are not printed unless this is True
    update_config_type: bool, default True
        append "MD" to at.info['config_type']
    selector_function: None
        Function to sub-select configs from the first trajectory. 
        Takes in list of configs and returns list of configs.

    Returns
    -------
        list(Atoms) trajectories
    """

    calculator = construct_calculator_picklesafe(calculator)

    all_trajs = []

    if verbose:
        logfile = '-'
    else:
        logfile = None

    if temperature_tau is None and not isinstance(temperature, float):
        raise RuntimeError('NVE (temperature_tau is None) can only accept temperature=float for initial T')

    if temperature is not None:
        if isinstance(temperature, (float, int)):
            # float into a list
            temperature = [temperature]
        if not isinstance(temperature[0], dict):
            # create a stage dict from a constant or ramp
            t_stage_data = temperature
            # start with constant
            t_stage = { 'T_i': t_stage_data[0], 'T_f' : t_stage_data[0], 'traj_frac': 1.0, 'n_stages': 10, 'steps': steps }
            if len(t_stage_data) >= 2:
                # set different final T for ramp
                t_stage['T_f'] = t_stage_data[1]
            if len(t_stage_data) >= 3:
                # set number of stages
                t_stage['n_stages'] = t_stage_data[2]
            temperature = [t_stage]
        else:
            for t_stage in temperature:
                if 'n_stages' not in t_stage:
                    t_stage['n_stages'] = 10

    for at in atoms_to_list(atoms):
        at.calc = calculator
        compressibility = None
        if pressure is not None:
            pressure = sample_pressure(pressure, at)
            at.info['MD_pressure_GPa'] = pressure
            # convert to ASE internal units
            pressure *= GPa

            E0 = at.get_potential_energy()
            c0 = at.get_cell()
            at.set_cell(c0 * (1.0 + compressibility_fd_displ), scale_atoms=True)
            Ep = at.get_potential_energy()
            at.set_cell(c0 * (1.0 - compressibility_fd_displ), scale_atoms=True)
            Em = at.get_potential_energy()
            at.set_cell(c0, scale_atoms=True)
            d2E_dF2 = (Ep + Em - 2.0 * E0) / (compressibility_fd_displ ** 2)
            compressibility = at.get_volume() / d2E_dF2

        if temperature is not None:
            # set initial temperature
            MaxwellBoltzmannDistribution(at, temperature[0]['T_i'] * kB, force_temp=True, communicator=None)
            Stationary(at, preserve_temperature=True)

        stage_kwargs = {'timestep': dt * fs, 'logfile': logfile}

        if temperature_tau is None:
            # NVE
            if pressure is not None:
                raise RuntimeError('Cannot do NPH dynamics')
            md_constructor = VelocityVerlet
            # one stage, simple
            all_stage_kwargs = [stage_kwargs.copy()]
            all_run_kwargs = [ {'steps': steps} ]
        else:
            # NVT or NPT
            all_stage_kwargs = []
            all_run_kwargs = []

            stage_kwargs['taut'] = temperature_tau * fs

            if pressure is not None:
                md_constructor = NPTBerendsen
                stage_kwargs['pressure_au'] = pressure
                stage_kwargs['compressibility_au'] = compressibility
                stage_kwargs['taup'] = temperature_tau * fs * 3 if pressure_tau is None else pressure_tau * fs
            else:
                md_constructor = NVTBerendsen

            for t_stage_i, t_stage in enumerate(temperature):
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
        bad_geometry_counter = 0
        previous_step_geometry_is_ok = True

        def process_step(interval):
            nonlocal cur_step, first_step_of_later_stage
            nonlocal bad_geometry_counter, previous_step_geometry_is_ok

            if not first_step_of_later_stage and cur_step % interval == 0:
                at.info['MD_time_fs'] = cur_step * dt
                traj.append(at_copy_save_results(at, results_prefix=results_prefix))
                geometry_ok = configs.check_geometry(at)
                if not geometry_ok:
                    bad_geometry_counter += 1
                    if bad_geometry_counter == 10 and not previous_step_geometry_is_ok:
                        raise RuntimeError("Had 10 bad geometry configs in a row")
                    previous_step_geometry_is_ok = False 
                else:
                    previous_step_geometry_is_ok = True
                    bad_geometry_counter = 0

            first_step_of_later_stage = False
            cur_step += 1

        for stage_i, (stage_kwargs, run_kwargs) in enumerate(zip(all_stage_kwargs, all_run_kwargs)):
            if verbose:
                print('run stage', stage_kwargs, run_kwargs)

            # avoid double counting of steps and end of each stage and beginning of next
            cur_step -= 1

            if temperature_tau is not None:
                at.info['MD_temperature_K'] = stage_kwargs['temperature_K']

            md = md_constructor(at, **stage_kwargs)
            md.attach(process_step, 1, traj_step_interval)

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
            at.info['MD_time_fs'] = cur_step * dt
            traj.append(at_copy_save_results(at, results_prefix=results_prefix))

        if update_config_type:
            # save config_type
            for at in traj:
                config_type_append(at, 'MD')

        if selector_function is not None:
            traj = selector_function(traj)

        all_trajs.append(traj)
        
    return all_trajs
