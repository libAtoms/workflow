import os
import shutil
import numpy as np
from pytest import approx
import pytest
from ase import Atoms
from ase.build import bulk
import wfl.calculators.castep
from wfl.configset import ConfigSet, OutputSpec
from wfl.autoparallelize import AutoparaInfo
from wfl.calculators import generic

pytestmark = pytest.mark.skipif("CASTEP_COMMAND" not in os.environ, reason="'CASTEP_COMMAND' not given.")

def test_castep_calculation(tmp_path):

    atoms = bulk("Al", "bcc", a=4.05, cubic=True) 

    castep_kwargs = {
        'write_checkpoint':"none",
        'cut_off_energy':400,
        'calculate_stress': True,
        'kpoints_mp_spacing': 0.04
            }

    calc = wfl.calculators.castep.Castep(
        workdir=tmp_path,
        **castep_kwargs
    )

    atoms.calc = calc
    assert atoms.get_potential_energy() == approx(-217.2263559019, 2e-3)
    assert atoms.get_forces() == approx(np.array([[-0., -0.,  0.], [ 0.,  0., -0.]]), abs=1e-4)
    assert atoms.get_stress() ==  approx(np.array([ 0.06361731, 0.06361731, 0.06361731,-0., 0., 0.]), abs=1e-5)


def test_castep_calc_via_generic(tmp_path):

    atoms = bulk("Al", "bcc", a=4.05, cubic=True) 
    cfgs = [atoms] 

    kwargs = {
        'write_checkpoint':"none",
        'cut_off_energy':400,
        'calculate_stress': True,
        'kpoints_mp_spacing': 0.04,
        'workdir' : tmp_path,
            }

    calc = (wfl.calculators.castep.Castep, [], kwargs)

    ci = ConfigSet(cfgs)
    co = OutputSpec()
    autoparainfo = AutoparaInfo(
        num_python_subprocesses=0
    )

    ci = generic.calculate(
        inputs=ci,
        outputs=co, 
        calculator=calc, 
        output_prefix='castep_',
        autopara_info=autoparainfo
    )



