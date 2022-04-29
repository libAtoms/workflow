from pytest import approx
import pytest

import numpy as np

from ase import Atoms
import ase.io
from ase.build import bulk
from ase.calculators.emt import EMT

from wfl.generate_configs import md
from wfl.configset import ConfigSet, OutputSpec


@pytest.fixture
def cu_slab():

    atoms = bulk("Cu", "fcc", a=3.6, cubic=True)
    atoms *= (2, 2, 2)
    atoms.rattle(stdev=0.01, seed=159)

    atoms.info['config_type'] = 'cu_slab'
    atoms.info['buildcell_config_i'] = 'fake_buildecell_config_name'

    return atoms


def test_NVE(cu_slab):
    calc = EMT()

    inputs = ConfigSet(input_configs = cu_slab)
    outputs = OutputSpec()

    atoms_traj = md.sample(inputs, outputs, calculator=calc, steps=300, dt=1.0,
                           temperature = 500.0)

    atoms_traj = list(atoms_traj)
    atoms_final = atoms_traj[-1]

    assert len(atoms_traj) == 301


def test_NVT_const_T(cu_slab):

    calc = EMT()

    inputs = ConfigSet(input_configs = cu_slab)
    outputs = OutputSpec()

    atoms_traj = md.sample(inputs, outputs, calculator=calc, steps=300, dt=1.0,
                           temperature = 500.0, temperature_tau=30.0)

    atoms_traj = list(atoms_traj)
    atoms_final = atoms_traj[-1]

    assert len(atoms_traj) == 301
    assert all([at.info['MD_temperature_K'] == 500.0 for at in atoms_traj])


def test_NVT_simple_ramp(cu_slab):

    calc = EMT()

    inputs = ConfigSet(input_configs = cu_slab)
    outputs = OutputSpec()

    atoms_traj = md.sample(inputs, outputs, calculator=calc, steps=300, dt=1.0,
                           temperature = (500.0, 100.0), temperature_tau=30.0)

    atoms_traj = list(atoms_traj)
    atoms_final = atoms_traj[-1]

    assert len(atoms_traj) == 301
    Ts = []
    for T_i, T in enumerate(np.linspace(500.0, 100.0, 10)):
        Ts.extend([T] * 30)
        if T_i == 0:
            Ts.append(T)
    assert all(np.isclose(Ts, [at.info['MD_temperature_K'] for at in atoms_traj]))


def test_NVT_complex_ramp(cu_slab):

    calc = EMT()

    inputs = ConfigSet(input_configs = cu_slab)
    outputs = OutputSpec()

    atoms_traj = md.sample(inputs, outputs, calculator=calc, steps=300, dt=1.0,
                           temperature = [{'T_i': 100.0, 'T_f': 500.0, 'traj_frac': 0.5},
                                          {'T_i': 500.0, 'T_f': 500.0, 'traj_frac': 0.25},
                                          {'T_i': 500.0, 'T_f': 300.0, 'traj_frac': 0.25}],
                           temperature_tau=30.0)

    atoms_traj = list(atoms_traj)
    atoms_final = atoms_traj[-1]

    assert len(atoms_traj) == 306
    Ts = []
    for T_i, T in enumerate(np.linspace(100.0, 500.0, 10)):
        Ts.extend([T] * 15)
        if T_i == 0:
            Ts.append(T)
    Ts.extend([500.0] * 75)
    for T_i, T in enumerate(np.linspace(500.0, 300.0, 10)):
        Ts.extend([T] * 8)

    # for at_i, at in enumerate(atoms_traj):
        # print(at_i, at.info['MD_time_fs'], 'MD', at.info['MD_temperature_K'], 'test', Ts[at_i])

    assert all(np.isclose(Ts, [at.info['MD_temperature_K'] for at in atoms_traj]))
