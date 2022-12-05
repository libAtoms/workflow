import numpy as np
from pytest import approx
from ase import Atoms
from ase.build import bulk
import wfl.calculators.castep


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
        calculator_exec = "mpirun -n 32 castep.mpi",
        **castep_kwargs
    )

    atoms.calc = calc
    assert atoms.get_potential_energy() == approx(-217.2263559019)
    assert atoms.get_forces() == approx(np.array([[-0., -0.,  0.], [ 0.,  0., -0.]]))
    assert atoms.get_stress() ==  approx(np.array([ 0.06361731, 0.06361731, 0.06361731,-0., 0., 0.]))

