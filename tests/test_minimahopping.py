import pytest
import numpy as np
from pathlib import Path 

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

def test_return_md_traj(cu_slab, tmp_path):
    
    calc = EMT()
    
    input_configs = cu_slab
    inputs = ConfigSet(input_configs)
    outputs = OutputSpec()
 
    atoms_opt = minimahopping.minimahopping(inputs, outputs, calc, fmax=1, totalsteps=5, save_tmpdir=True, return_all_traj=True,
                                            rng=np.random.default_rng(1), workdir=tmp_path)

    assert any(["minhop_min" in at.info["config_type"] for at in atoms_opt])
    assert any(["minhop_traj" in at.info["config_type"] for at in atoms_opt])


def test_mult_files(cu_slab, tmp_path):
    ase.io.write(tmp_path / 'f1.xyz', [cu_slab] * 2)
    ase.io.write(tmp_path / 'f2.xyz', cu_slab)
    infiles = [str(tmp_path / 'f1.xyz'), str(tmp_path / 'f2.xyz')]
    inputs = ConfigSet(infiles)
    outputs = OutputSpec([f.replace('.xyz', '.out.xyz') for f in infiles])

    calc = EMT()

    atoms_opt = minimahopping.minimahopping(inputs, outputs, calc, fmax=1, totalsteps=3, 
                                            rng=np.random.default_rng(1), workdir=tmp_path)

    n1 = len(ase.io.read(tmp_path / infiles[0].replace('.xyz', '.out.xyz'), ':'))
    n2 = len(ase.io.read(tmp_path / infiles[1].replace('.xyz', '.out.xyz'), ':'))

    assert n1 == n2 * 2


def test_relax(cu_slab, tmp_path):

    calc = EMT()

    input_configs = [cu_slab, cu_slab]
    inputs = ConfigSet(input_configs)
    outputs = OutputSpec()

    fmax = 1
    totalsteps = 3
    num_input_configs = len(input_configs)

    # Although its highly unlikely, we can't fully guarantee that the situation
    # where are trajectories fail is excluded. Thus, let's give it some trials to avoid this situation.
    trial = 0
    while trial < 10:
        atoms_opt = minimahopping.minimahopping(inputs, outputs, calc, fmax=fmax, totalsteps=totalsteps,
                                                rng=np.random.default_rng(1), workdir=tmp_path)
        if len(list(atoms_opt)) > 0:
            break
        trial += 1
    else:
        raise RuntimeError

    assert 1 <= len(list(atoms_opt)) <= num_input_configs
    for ats in atoms_opt.groups():
        assert 1 <= len(list(ats)) <= totalsteps

    atoms_opt = list(atoms_opt)
    assert all(['minhop_min' in at.info['config_type'] for at in atoms_opt])

    for at in atoms_opt:
        force_norms = np.linalg.norm(at.arrays["last_op__minhop_forces"], axis=1)
        assert all(force_norms <= fmax)
