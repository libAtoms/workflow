from pytest import approx
import pytest

import numpy as np

from ase import Atoms
import ase.io
from ase.build import bulk
from ase.build import add_adsorbate, fcc100
from ase.calculators.emt import EMT

from ase.lattice.cubic import FaceCenteredCubic
from ase.mep import DyNEB
from ase.optimize.fire import FIRE as QuasiNewton
from ase.constraints import FixAtoms

#from wfl.generate.neb import _run_autopara_wrappable # NEB as wflNEB
#from wfl.generate.neb import NEB as wflNEB
from wfl.generate import neb 
from wfl.autoparallelize.autoparainfo import AutoparaInfo
from wfl.configset import ConfigSet, OutputSpec


@pytest.fixture
def prepare_images():

    # Set the number of images you want.
    nimages = 5
    
    # Some algebra to determine surface normal and the plane of the surface.
    d3 = [2, 1, 1]
    a1 = np.array([0, 1, 1])
    d1 = np.cross(a1, d3)
    a2 = np.array([0, -1, 1])
    d2 = np.cross(a2, d3)
    
    # Create the slab.
    slab = FaceCenteredCubic(directions=[d1, d2, d3],
                             size=(2, 1, 2),
                             symbol=('Pt'),
                             latticeconstant=3.9)
    
    # Add some vacuum to the slab.
    uc = slab.cell
    uc[2] += [0., 0., 10.]  # There are ten layers of vacuum.
    slab.set_cell(uc, scale_atoms=False)
    
    # Some positions needed to place the atom in the correct place.
    x1 = 1.379
    x2 = 4.137
    x3 = 2.759
    y1 = 0.0
    y2 = 2.238
    z1 = 7.165
    z2 = 6.439
    
    # Add the adatom to the list of atoms and set constraints of surface atoms.
    slab += Atoms('N', [((x2 + x1) / 2, y1, z1 + 1.5)])
    mask = [atom.symbol == 'Pt' for atom in slab]
    slab.set_constraint(FixAtoms(mask=mask))
    
    # Optimise the initial state: atom below step.
    initial = slab.copy()
    initial.calc = EMT()
    relax = QuasiNewton(initial)
    relax.run(fmax=0.05)
    
    # Optimise the final state: atom above step.
    slab[-1].position = (x3, y2 + 1., z2 + 3.5)
    final = slab.copy()
    final.calc = EMT()
    relax = QuasiNewton(final)
    relax.run(fmax=0.05)

    initial.calc = None
    final.calc = None

    return initial, final


def test_neb(prepare_images):
    
    calc = (EMT, [], {})
    initial, final = prepare_images

    images = [initial.copy() for _ in range(6)] + [final.copy()]
    
    # Carry out idpp interpolation.
    dyn1 = DyNEB(images)
    dyn1.interpolate('idpp')

    inputs = ConfigSet([dyn1.images])
    outputs = OutputSpec()

    images_opt = neb.NEB(
        inputs.groups(),
		outputs,
        calc,
        logfile="-",
        verbose=True,
        allow_shared_calculator = True,
    )

    images_opt = list(images_opt)[-7:]

#    assert atoms_opt.positions == approx(
#        expected_relaxed_positions_constant_pressure, abs=3e-3
#    )

    assert np.all([at.info["config_type"] == "neb_last_converged" for at in images_opt])


def test_neb_autopara_full_mult_files(prepare_images, tmp_path):
    do_test_neb_autopara_full(prepare_images, tmp_path,
                              [tmp_path / 'out_1.xyz', tmp_path / 'out_2.xyz'])


def test_neb_autopara_full_one_file(prepare_images, tmp_path):
    do_test_neb_autopara_full(prepare_images, tmp_path, tmp_path / 'out.xyz')


def test_neb_autopara_last_mult_files(prepare_images, tmp_path):
    do_test_neb_autopara_last(prepare_images, tmp_path,
                              [tmp_path / 'out_1.xyz', tmp_path / 'out_2.xyz'])


def test_neb_autopara_last_one_file(prepare_images, tmp_path):
    do_test_neb_autopara_last(prepare_images, tmp_path, tmp_path / 'out.xyz')


def do_test_neb_autopara_full(prepare_images, tmp_path, output_files):

    calc = (EMT, [], {})
    initial, final = prepare_images

    images1 = [initial.copy() for _ in range(6)] + [final.copy()]
    images2 = [initial.copy() for _ in range(6)] + [final.copy()]

    # Carry out idpp interpolation.
    dyn1 = DyNEB(images1)
    dyn1.interpolate('idpp')
    dyn2 = DyNEB(images2)
    dyn2.interpolate("idpp")

    # this one writes to multiple files
    inputs = ConfigSet([dyn1.images, dyn2.images])
    outputs = OutputSpec(output_files)

    images_opt = neb.NEB(
        inputs.groups(),
        outputs,
        calc,
        autopara_info=AutoparaInfo(num_python_subprocesses=2,
                                   num_inputs_per_python_subprocess=1),
        logfile="-",
        verbose=True,
        allow_shared_calculator=True,
    )

    images_opt = list(images_opt)[-7:]

#    assert atoms_opt.positions == approx(
#        expected_relaxed_positions_constant_pressure, abs=3e-3
#    )

    assert np.all([at.info["config_type"] == "neb_last_converged" for at in images_opt])


def do_test_neb_autopara_last(prepare_images, tmp_path, output_files):

    calc = (EMT, [], {})
    initial, final = prepare_images

    images1 = [initial.copy() for _ in range(6)] + [final.copy()]
    images2 = [initial.copy() for _ in range(6)] + [final.copy()]

    # Carry out idpp interpolation.
    dyn1 = DyNEB(images1)
    dyn1.interpolate('idpp')
    dyn2 = DyNEB(images2)
    dyn2.interpolate("idpp")

    # this one writes to one files
    inputs = ConfigSet([dyn1.images, dyn2.images])
    outputs = OutputSpec(output_files)

    images_opt = neb.NEB(
        inputs.groups(),
        outputs,
        calc,
        autopara_info=AutoparaInfo(num_python_subprocesses=2,
                                   num_inputs_per_python_subprocess=1),
        logfile="-",
        verbose=True,
        allow_shared_calculator=True,
        traj_subselect="last",
    )

    images_opt = list(images_opt)[-7:]

#    assert atoms_opt.positions == approx(
#        expected_relaxed_positions_constant_pressure, abs=3e-3
#    )

    assert np.all([at.info["config_type"] == "neb_last_converged" for at in images_opt])
