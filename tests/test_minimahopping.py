import pytest

from ase import Atoms
import ase.io
from ase.build import bulk
from ase.calculators.emt import EMT

from wfl.generate import minimahopping
from wfl.configset import ConfigSet, OutputSpec

@pytest.fixture
def cu_slab():

    atoms = bulk("Cu", "fcc", a=3.6, cubic=True)
    atoms *= (2, 2, 2)
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

    inputs = ConfigSet(input_configs = cu_slab)
    outputs = OutputSpec()

    atoms_opt = minimahopping.run(inputs, outputs, calc, fmax=1, totalsteps=3)
    
    atoms_opt = list(atoms_opt)[-1]

    assert atoms_opt.info['config_type'] == 'hopping_traj'

