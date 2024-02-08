from pytest import approx
import pytest

import numpy as np

from ase import Atoms
import ase.io
from ase.build import bulk
from ase.calculators.emt import EMT

from ase.constraints import FixAtoms

from wfl.generate import optimize
from wfl.configset import ConfigSet, OutputSpec

expected_relaxed_positions_constant_pressure = np.array(
    [
        [7.64860000e-04, 6.66750000e-04, 1.12750000e-04],
        [-6.24100000e-05, 1.80043870e00, 1.79974967e00],
        [1.80032231e00, -2.97400000e-05, 1.79973549e00],
        [1.80095487e00, 1.80017670e00, 2.61750000e-04],
        [1.15840000e-03, 2.36710000e-04, 3.59941539e00],
        [1.48600000e-03, 1.79993916e00, 5.40024293e00],
        [1.80072223e00, -4.23990000e-04, 5.39933992e00],
        [1.80046609e00, 1.80061120e00, 3.59922853e00],
        [4.40530000e-04, 3.59983780e00, 5.02600000e-05],
        [-2.23490000e-04, 5.40075346e00, 1.79980812e00],
        [1.80038005e00, 3.59997803e00, 1.79959318e00],
        [1.79921287e00, 5.40012037e00, 1.56140000e-04],
        [1.99000000e-04, 3.60031722e00, 3.60034182e00],
        [2.86780000e-04, 5.40104179e00, 5.39939418e00],
        [1.80032597e00, 3.60108108e00, 5.39968833e00],
        [1.80021004e00, 5.40062669e00, 3.60019918e00],
        [3.60087182e00, -3.65010000e-04, 7.61500000e-04],
        [3.60073516e00, 1.80048495e00, 1.80054603e00],
        [5.40067545e00, -1.13950000e-04, 1.79960057e00],
        [5.39941457e00, 1.79967203e00, -1.98190000e-04],
        [3.60059144e00, -4.27610000e-04, 3.59954728e00],
        [3.59999268e00, 1.79976207e00, 5.39918637e00],
        [5.40022858e00, 9.34800000e-05, 5.39977362e00],
        [5.40030500e00, 1.79994095e00, 3.59992025e00],
        [3.60059773e00, 3.60070484e00, 3.26930000e-04],
        [3.60013298e00, 5.39976537e00, 1.79997636e00],
        [5.40026903e00, 3.59927606e00, 1.80032592e00],
        [5.40033996e00, 5.40040411e00, 8.72140000e-04],
        [3.60095555e00, 3.60037761e00, 3.60000981e00],
        [3.60089465e00, 5.40081989e00, 5.40020294e00],
        [5.39998633e00, 3.59971275e00, 5.39954472e00],
        [5.40027502e00, 5.40112125e00, 3.59984043e00],
    ]
)


@pytest.fixture
def cu_slab():

    atoms = bulk("Cu", "fcc", a=3.6, cubic=True)
    atoms *= (2, 2, 2)
    atoms.rattle(stdev=0.01, seed=159)

    atoms.info["config_type"] = "cu_slab"
    atoms.info["buildcell_config_i"] = "fake_buildecell_config_name"

    return atoms


def test_mult_files(cu_slab, tmp_path):
    ase.io.write(tmp_path / 'f1.xyz', [cu_slab] * 2)
    ase.io.write(tmp_path / 'f2.xyz', cu_slab)
    infiles = [str(tmp_path / 'f1.xyz'), str(tmp_path / 'f2.xyz')]
    inputs = ConfigSet(infiles)
    outputs = OutputSpec([f.replace('.xyz', '.out.xyz') for f in infiles])

    calc = EMT()

    atoms_opt = optimize.optimize(
        inputs,
        outputs,
        calc,
        fmax=1e-2,
        precon=None,
        logfile="-",
        verbose=True,
        pressure=-1.1215,
    )

    n1 = len(ase.io.read(tmp_path / infiles[0].replace(".xyz", ".out.xyz"), ":"))
    n2 = len(ase.io.read(tmp_path / infiles[1].replace(".xyz", ".out.xyz"), ":"))

    assert n1 == n2 * 2


def test_relax(cu_slab):

    calc = EMT()

    inputs = ConfigSet(cu_slab)
    outputs = OutputSpec()

    atoms_opt = optimize.optimize(
        inputs,
        outputs,
        calc,
        fmax=1e-2,
        precon=None,
        logfile="-",
        verbose=True,
        pressure=-1.1215,
    )

    atoms_opt = list(atoms_opt)[-1]

    print("optimised positions:", atoms_opt.positions)

    assert atoms_opt.positions == approx(
        expected_relaxed_positions_constant_pressure, abs=3e-3
    )

    assert atoms_opt.info["config_type"] == "cu_slab_optimize_last_converged"


def test_relax_fixed_vol(cu_slab):

    calc = EMT()

    inputs = ConfigSet(cu_slab)
    outputs = OutputSpec()

    atoms_opt = optimize.optimize(
        inputs, outputs, calc, fmax=1e-2, precon=None, logfile="-", verbose=True
    )

    atoms_opt = list(atoms_opt)[-1]

    # import json, sys
    # json.dump(atoms_opt.positions.tolist(), sys.stdout)

    expected_positions = np.asarray(
        [
            [0.0006051702533344901, 0.00045597443454537, 6.327123097568655e-05],
            [0.00012399319222662348, 1.7998299507660063, 1.7997665575003574],
            [1.8004406297943782, -0.00010537880138260529, 1.8000740002698472],
            [1.8008911772213716, 1.8002693883257999, 0.0002511815147499709],
            [0.0007476594240690664, 5.8848857966311294e-05, 3.5999258439149893],
            [0.0008526218177704321, 1.8000635491169206, 5.400060213622148],
            [1.800861471132415, -7.919071942340348e-05, 5.399975928241347],
            [1.8003584859877908, 1.8000219205333048, 3.5999538876513246],
            [0.0003345377959548103, 3.599967621326831, -0.00017201455070664365],
            [0.00018336225261436016, 5.399727679789856, 1.799877388997259],
            [1.7999859084512086, 3.5997522140437694, 1.7998981411726311],
            [1.8002699793915604, 5.400022785876646, -0.00020664866198387172],
            [7.737695163795191e-05, 3.60001023626587, 3.6000604377499004],
            [0.00036661638791551585, 5.400359205718571, 5.399805852753726],
            [1.800527138745017, 3.6003537912994212, 5.400109681578851],
            [1.8002973726356255, 5.3999497646418, 3.599924043824988],
            [3.6006680969672784, -0.00020717275611322345, 0.0001387434122933971],
            [3.600211961517994, 1.7999763151128534, 1.8003262136034315],
            [5.400573975423159, -2.905058430219612e-05, 1.799589911628563],
            [5.400211892111432, 1.7999642754577954, -0.00040352564722177277],
            [3.600371202951032, -0.00016321333195181906, 3.599737740181237],
            [3.600258784568573, 1.799865939862209, 5.39987542352589],
            [5.400658225501837, 0.00031484448131756414, 5.399832790089358],
            [5.400361858161337, 1.7998328631998441, 3.6001292700868297],
            [3.600379196067282, 3.6001803304064697, -5.294834218475312e-06],
            [3.6003729181078303, 5.399558512498043, 1.799775338872422],
            [5.400066789585735, 3.5998171577348126, 1.7999294874216605],
            [5.40079809470619, 5.4001164890863285, -0.000173024271889441],
            [3.6005363741779393, 3.600110600005077, 3.599989189954323],
            [3.6006361037257677, 5.400222837357979, 5.400030167838016],
            [5.400461842162823, 3.599913633723715, 5.399591046034773],
            [5.400425465606757, 5.400014020338249, 3.599479942759126],
        ]
    )

    print("optimised positions:", atoms_opt.positions)

    assert atoms_opt.positions == approx(expected_positions, abs=3e-3)

    assert atoms_opt.info["config_type"] == "cu_slab_optimize_last_converged"


def test_subselect_from_traj(cu_slab):

    calc = (EMT, [], {})

    cu_slab_optimised = cu_slab.copy()
    cu_slab_optimised.set_positions(expected_relaxed_positions_constant_pressure)

    # returns full optimisation trajectories
    inputs = [cu_slab.copy(), cu_slab_optimised.copy()]
    atoms_opt = optimize._run_autopara_wrappable(
        inputs,
        calc,
        fmax=1e-2,
        precon=None,
        logfile="-",
        verbose=True,
        pressure=-1.1215,
        steps=2,
        traj_subselect=None,
        _autopara_per_item_info = [{} for _ in range(len(inputs))]
    )

    assert len(atoms_opt[0]) == 3
    assert len(atoms_opt[1]) == 1
    assert isinstance(atoms_opt[0][0], Atoms)
    assert isinstance(atoms_opt[1][0], Atoms)

    # returns [None] for unconverged and last config for converged otpimisation
    inputs = [cu_slab.copy(), cu_slab_optimised.copy()]
    atoms_opt = optimize._run_autopara_wrappable(
        inputs,
        calc,
        fmax=1e-2,
        precon=None,
        logfile="-",
        verbose=True,
        pressure=-1.1215,
        steps=2,
        traj_subselect="last_converged",
        _autopara_per_item_info = [{} for _ in range(len(inputs))]
    )

    assert atoms_opt[0] is None
    assert isinstance(atoms_opt[1], Atoms)  # not None

    # check that iterable_loop handles Nones as expected
    inputs = ConfigSet([cu_slab.copy(), cu_slab_optimised.copy()])
    outputs = OutputSpec()
    atoms_opt = optimize.optimize(
        inputs,
        outputs,
        calc,
        fmax=1e-2,
        precon=None,
        logfile="-",
        verbose=True,
        pressure=-1.1215,
        steps=2,
        traj_subselect="last_converged",
    )

    atoms_opt = list(atoms_opt)
    assert len(atoms_opt) == 1
    assert isinstance(atoms_opt[0], Atoms)


def test_relax_with_constraints(cu_slab):
    """
    Test relaxation with FixAtoms constraint, when keep_symetry = True

    Test 1) Wether Fixed atoms stay fixed 2) wether the constraints are the same
    at the end of the relaxation eg. that the symetry constraint is removed

    Args:
        cu_slab (ase.Atoms): test configuration
    """
    calc = EMT()
    ats = cu_slab

    # Fix lowest 10 atoms
    pos = ats.get_positions()
    fix_indices = np.where(pos[:, 2] < np.sort(pos[:, 2])[min(10, len(ats))])[0]
    c = FixAtoms(indices=fix_indices)
    ats.set_constraint(c)

    # Save copy of constraints to compre later
    org_constraints = ats.constraints.copy()

    # Rattle such that the relaxation takes a few steps
    ats.rattle(0.1, seed=0)
    inputs = ConfigSet(ats)
    outputs = OutputSpec()

    atoms_opt = optimize.optimize(
        inputs, outputs, calc, fmax=1e-2, keep_symmetry=True, logfile="-", verbose=True
    )
    output = list(atoms_opt)

    # Changes in position
    diff_pos = output[0].get_positions() - output[-1].get_positions()
    assert (
        np.sum(diff_pos[fix_indices]) < 1e-10
    ), "Fixed atoms should not move during relaxation"

    assert len(output[-1].constraints) == len(
        org_constraints
    ), "Number of constraints on atoms should be the same before and after relaxation"
