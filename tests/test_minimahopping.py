import pytest
import numpy as np

import ase.io
from ase.build import bulk
from ase.calculators.emt import EMT

from wfl.generate import minimahopping
from wfl.configset import ConfigSet, OutputSpec


@pytest.fixture
def cu_slab():

    atoms = bulk("Cu", "fcc", a=3.8, cubic=True)
    atoms.rattle(stdev=0.01, seed=159)

    atoms.info['config_type'] = 'cu_slab'

    return atoms


def test_mult_files(cu_slab, tmp_path):
    ase.io.write(tmp_path / 'f1.xyz', [cu_slab] * 2)
    ase.io.write(tmp_path / 'f2.xyz', cu_slab)
    infiles = [str(tmp_path / 'f1.xyz'), str(tmp_path / 'f2.xyz')]
    inputs = ConfigSet(input_files=infiles)
    outputs = OutputSpec(output_files={f: f.replace('.xyz', '.out.xyz') for f in infiles})

    calc = EMT()

    atoms_opt = minimahopping.run(inputs, outputs, calc, fmax=1, totalsteps=3)

    n1 = len(ase.io.read(tmp_path / infiles[0].replace('.xyz', '.out.xyz'), ':'))
    n2 = len(ase.io.read(tmp_path / infiles[1].replace('.xyz', '.out.xyz'), ':'))

    assert n1 == n2 * 2


def test_relax(cu_slab):

    calc = EMT()

    input_configs = [cu_slab, cu_slab]
    inputs = ConfigSet(input_configs=input_configs)
    outputs = OutputSpec()

    fmax = 1
    totalsteps = 3
    num_input_configs = len(input_configs)

    # Although its highly unlikely, we can't fully guarantee that the situation
    # where are trajectories fail is excluded. Thus, let's give it some trials to avoid this situation.
    # Failed trajectories for which None is returned are filtered out by the
    # autoparallelize framework. In case only None values are returned the autoparallelize-framework
    # returns an empty ConfigSet with self.input_configs = None, which consequently has no length
    # and can't be tested properly.
    trial = 0
    while trial < 10:
        atoms_opt = minimahopping.run(inputs, outputs, calc, fmax=fmax, totalsteps=totalsteps)
        if atoms_opt.input_configs is not None:
            break
        trial += 1
    else:
        raise RuntimeError

    assert 1 <= len(atoms_opt.input_configs) <= num_input_configs
    for ats in atoms_opt.input_configs:
        assert 1 <= len(ats) <= totalsteps

    atoms_opt = list(atoms_opt)
    assert all([at.info['config_type'] == 'hopping_traj' for at in atoms_opt])

    for at in atoms_opt:
        force_norms = np.linalg.norm(at.get_forces(), axis=1)
        assert all(force_norms <= fmax)
