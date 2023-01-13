import os
import pytest

import numpy as np
from ase import Atoms
from ase.build import bulk

import wfl.calculators.aims
from wfl.calculators import generic
from wfl.configset import OutputSpec


aims_prerequisites = pytest.mark.skipif(
        condition='ASE_AIMS_COMMAND' not in os.environ or 'AIMS_SPECIES_DIR' not in os.environ
                   or not os.environ['AIMS_SPECIES_DIR'].endswith('light')
                   or 'OMP_NUM_THREADS' not in os.environ or os.environ['OMP_NUM_THREADS'] != "1",
        reason='Missing env var ASE_AIMS_COMMAND or ASE_SPECIES_DIR or ' +
               'ASE_SPECIES_DIR does not refer to light-settings or ' +
               'OMP_NUM_THREADS or OMP_NUM_THREADS has not correctly been set to 1'
        )


@pytest.fixture
def parameters_nonperiodic():
    parameters = {
        'xc': 'pbe',
        'spin': 'none',
        'relativistic': 'none',
        'charge': 0.,
        'sc_iter_limit': 500,
        'occupation_type': 'gaussian 0.01',
        'charge_mix_param': 0.6,
        'mixer': 'pulay',
        'n_max_pulay': 10,
        'sc_accuracy_rho': 1e-2,
        'sc_accuracy_eev': 1e-2,
        'sc_accuracy_etot': 1e-4,
        'sc_accuracy_forces': 1E-2,
        'compute_forces': True,
        'KS_method': 'parallel',
    }
    return parameters


def test_setup_calc_params(parameters_nonperiodic):

    parameters = parameters_nonperiodic
    parameters_periodic = {
        'k_grid': '1 1 1',
        'k_grid_density': 0.1,
        'k_offset': 0.1,
        'relax_unit_cell': 'full',
        'external_pressure': 10,
        'compute_analytical_stress': '.true.',
        'sc_accuracy_stress': 1e-1,
    }
    parameters.update(parameters_periodic)

    # PBC is FFF
    atoms = Atoms("H")
    calc = wfl.calculators.aims.Aims(**parameters)
    calc.atoms = atoms.copy()
    properties = ["energy", "forces", "stress"]
    properties = calc._setup_calc_params(properties)

    assert properties == ["energy", "forces"]
    for key_i in parameters_periodic.keys():
        assert key_i not in calc.parameters

    # PBC is TTT
    atoms = Atoms("H", cell=[1, 1, 1], pbc=True)
    calc = wfl.calculators.aims.Aims(**parameters)
    calc.atoms = atoms.copy()
    properties = ["energy", "forces", "stress"]
    properties = calc._setup_calc_params(properties)

    assert properties == ["energy", "forces", "stress"]
    for key_i in parameters_periodic.keys():
        assert key_i in calc.parameters


@aims_prerequisites
def test_aims_calculation(tmp_path, parameters_nonperiodic):

    atoms = Atoms("Si", cell=(2, 2, 2), pbc=[True] * 3)
    parameters = parameters_nonperiodic
    parameters.update({'k_grid': '1 1 1', 'compute_analytical_stress': '.true.'})

    calc = wfl.calculators.aims.Aims(
        workdir=tmp_path,
        **parameters)
    atoms.calc = calc

    atoms.get_potential_energy()
    atoms.get_forces()
    atoms.get_stress()


@aims_prerequisites
def test_generic_aims_calculation(tmp_path, parameters_nonperiodic):

    # atoms
    at = bulk("Si")
    at.positions[0, 0] += 0.01
    at0 = Atoms("Si", cell=[6.0, 6.0, 6.0], positions=[[3.0, 3.0, 3.0]], pbc=False)

    kw = parameters_nonperiodic
    kw.update({'k_grid': '1 1 1', 'compute_analytical_stress': '.true.', 'workdir': tmp_path})

    calc = (wfl.calculators.aims.Aims, [], kw)

    # output container
    c_out = OutputSpec("aims_results.xyz", file_root=tmp_path)

    results = generic.run(
        inputs=[at0, at],
        outputs=c_out,
        calculator=calc,
        output_prefix='Aims_',
    )

    # unpack the configset
    si_single, si2 = list(results)

    # dev: type hints
    si_single: Atoms
    si2: Atoms

    # single atoms tests
    assert "Aims_energy" in si_single.info
    assert "Aims_stress" not in si_single.info
    assert si_single.info["Aims_energy"] == pytest.approx(expected=-7869.54379922653, abs=1e-2)
    assert si_single.get_volume() == pytest.approx(6.0 ** 3)

    # bulk Si tests
    assert "Aims_energy" in si2.info
    assert "Aims_stress" in si2.info
    assert si2.info["Aims_energy"] == pytest.approx(expected=-15733.0158245551, abs=1e-3)
    assert si2.info["Aims_stress"] == pytest.approx(
        expected=np.array([-0.2813842, -0.2814739, -0.2814739, -0.00287345,  0., 0.]),
        abs=1e-3,
    )
    assert "Aims_forces" in si2.arrays
    assert si2.arrays["Aims_forces"][0, 0] == pytest.approx(expected=-0.29253217, abs=1e-3)
    assert si2.arrays["Aims_forces"][:, 1:] == pytest.approx(0.0)
    assert si2.arrays["Aims_forces"][0] == pytest.approx(-1 * si2.arrays["Aims_forces"][1])
