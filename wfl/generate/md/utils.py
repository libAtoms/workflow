import numpy as np

from ase.units import GPa

from wfl.utils.pressure import sample_pressure

def _get_temperature(temperature_use, temperature_tau, steps):
    if temperature_tau is None and (temperature_use is not None and not isinstance(temperature_use, (float, int, np.floating, np.integer))):
        raise RuntimeError(f'NVE (temperature_tau is None) can only accept temperature=float for initial T, got {type(temperature_use)}')

    if temperature_use is not None:
        # assume that dicts are already in temperature profile format
        if not isinstance(temperature_use, dict):
            try:
                # check if it's a list, tuple, etc
                len(temperature_use)
            except TypeError:
                # number into a list
                temperature_use = [temperature_use]
        if not isinstance(temperature_use[0], dict):
            # create a stage dict from a constant or ramp
            t_stage_data = temperature_use
            # start with constant
            t_stage = {'T_i': t_stage_data[0], 'T_f': t_stage_data[0], 'traj_frac': 1.0, 'n_stages': 10, 'steps': steps}
            if len(t_stage_data) >= 2:
                # set different final T for ramp
                t_stage['T_f'] = t_stage_data[1]
            if len(t_stage_data) >= 3:
                # set number of stages
                t_stage['n_stages'] = t_stage_data[2]
            temperature_use = [t_stage]
        else:
            for t_stage in temperature_use:
                if 'n_stages' not in t_stage:
                    t_stage['n_stages'] = 10

    return temperature_use

def _get_pressure(pressure_use, compressibility_au_use, compressibility_fd_displ, at, rng):
    if pressure_use is not None:
        pressure_use = sample_pressure(pressure_use, at, rng=rng)
        at.info['MD_pressure_GPa'] = pressure_use
        # convert to ASE internal units
        pressure_use *= GPa
        if compressibility_au_use is None:
            E0 = at.get_potential_energy()
            c0 = at.get_cell()
            at.set_cell(c0 * (1.0 + compressibility_fd_displ), scale_atoms=True)
            Ep = at.get_potential_energy()
            at.set_cell(c0 * (1.0 - compressibility_fd_displ), scale_atoms=True)
            Em = at.get_potential_energy()
            at.set_cell(c0, scale_atoms=True)
            d2E_dF2 = (Ep + Em - 2.0 * E0) / (compressibility_fd_displ ** 2)
            compressibility_au_use = at.get_volume() / d2E_dF2

    return pressure_use, compressibility_au_use
