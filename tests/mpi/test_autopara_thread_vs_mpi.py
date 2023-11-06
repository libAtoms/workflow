# for this test to work WFL_MPIPOOL must be in os.environ, because
# wfl.mpipool_support.init() requires it to actually do something

import os
from pathlib import Path

import numpy as np
import pytest
from ase import Atoms
from ase.build import molecule
from ase.calculators.lj import LennardJones

import wfl.autoparallelize.mpipool_support
from wfl.calculators import generic
from wfl.configset import ConfigSet, OutputSpec


def get_atoms():
    atoms = [molecule("CH4").copy() for _ in range(1000)]
    rng = np.random.RandomState(5)
    for at in atoms:
        at.rattle(0.1, rng=rng)
    return atoms


@pytest.mark.skipif('WFL_MPIPOOL' not in os.environ,
                    reason="only if WFL_MPIPOOL is in env")
@pytest.mark.mpi(minsize=2)
def test_run(tmp_path):
    from mpi4py import MPI

    if MPI.COMM_WORLD.rank > 0:
        return

    ## assert MPI.COMM_WORLD.size > 1

    # on one thread, run a serial reference calc
    os.environ['WFL_NUM_PYTHON_SUBPROCESSES'] = '0'

    mol_in = get_atoms()

    serial_mol_out = generic.calculate(mol_in, OutputSpec(tmp_path / "run_serial.xyz"),
                                 LennardJones(),
                                 properties=["energy", "forces"], output_prefix="_auto_")
    # check that serial output is correct type of object
    assert isinstance(serial_mol_out, ConfigSet)
    for at in serial_mol_out:
        assert isinstance(at, Atoms)

    # re-enable mpi pool based parallelism (although actual value is ignore if > 0 )
    os.environ['WFL_NUM_PYTHON_SUBPROCESSES'] = str(MPI.COMM_WORLD.size)

    mol_in = get_atoms()

    mpi_mol_out = generic.calculate(mol_in, OutputSpec(tmp_path / "run_mpi.xyz"), LennardJones(),
                              properties=["energy", "forces"], output_prefix="_auto_")
    # check that MPI parallel output is correct type of object
    assert isinstance(mpi_mol_out, ConfigSet)
    for at in mpi_mol_out:
        assert isinstance(at, Atoms)

    # check that serial and MPI parallel outputs agree
    for at_t, at_m in zip(serial_mol_out, mpi_mol_out):
        # print(at_t == at_m, at_t.info['LennardJones_energy'], at_m.info['LennardJones_energy'])
        assert at_t == at_m


if __name__ == '__main__':
    import conftest
    conftest.do_init_mpipool()

    rundir = Path('.')
    test_run(rundir)
    try:
        os.unlink(rundir / 'run_serial.xyz')
    except FileNotFoundError:
        pass
    try:
        os.unlink(rundir / 'run_mpi.xyz')
    except FileNotFoundError:
        pass
