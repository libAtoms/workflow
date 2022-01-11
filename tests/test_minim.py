from pytest import approx
import pytest

import numpy as np

from ase import Atoms
import ase.io
from ase.build import bulk
from ase.calculators.emt import EMT

from wfl.generate_configs import minim
from wfl.configset import ConfigSet_in, ConfigSet_out

expected_relaxed_positions_constant_pressure = np.array(
    [[ 7.64860000e-04,  6.66750000e-04,  1.12750000e-04],
    [-6.24100000e-05,  1.80043870e+00,  1.79974967e+00],
    [ 1.80032231e+00, -2.97400000e-05,  1.79973549e+00],
    [ 1.80095487e+00,  1.80017670e+00,  2.61750000e-04],
    [ 1.15840000e-03,  2.36710000e-04,  3.59941539e+00],
    [ 1.48600000e-03,  1.79993916e+00,  5.40024293e+00],
    [ 1.80072223e+00, -4.23990000e-04,  5.39933992e+00],
    [ 1.80046609e+00,  1.80061120e+00,  3.59922853e+00],
    [ 4.40530000e-04,  3.59983780e+00,  5.02600000e-05],
    [-2.23490000e-04,  5.40075346e+00,  1.79980812e+00],
    [ 1.80038005e+00,  3.59997803e+00,  1.79959318e+00],
    [ 1.79921287e+00,  5.40012037e+00,  1.56140000e-04],
    [ 1.99000000e-04,  3.60031722e+00,  3.60034182e+00],
    [ 2.86780000e-04,  5.40104179e+00,  5.39939418e+00],
    [ 1.80032597e+00,  3.60108108e+00,  5.39968833e+00],
    [ 1.80021004e+00,  5.40062669e+00,  3.60019918e+00],
    [ 3.60087182e+00, -3.65010000e-04,  7.61500000e-04],
    [ 3.60073516e+00,  1.80048495e+00,  1.80054603e+00],
    [ 5.40067545e+00, -1.13950000e-04,  1.79960057e+00],
    [ 5.39941457e+00,  1.79967203e+00, -1.98190000e-04],
    [ 3.60059144e+00, -4.27610000e-04,  3.59954728e+00],
    [ 3.59999268e+00,  1.79976207e+00,  5.39918637e+00],
    [ 5.40022858e+00,  9.34800000e-05,  5.39977362e+00],
    [ 5.40030500e+00,  1.79994095e+00,  3.59992025e+00],
    [ 3.60059773e+00,  3.60070484e+00,  3.26930000e-04],
    [ 3.60013298e+00,  5.39976537e+00,  1.79997636e+00],
    [ 5.40026903e+00,  3.59927606e+00,  1.80032592e+00],
    [ 5.40033996e+00,  5.40040411e+00,  8.72140000e-04],
    [ 3.60095555e+00,  3.60037761e+00,  3.60000981e+00],
    [ 3.60089465e+00,  5.40081989e+00,  5.40020294e+00],
    [ 5.39998633e+00,  3.59971275e+00,  5.39954472e+00],
    [ 5.40027502e+00,  5.40112125e+00,  3.59984043e+00]])


@pytest.fixture
def cu_slab():

    atoms = bulk("Cu", "fcc", a=3.6, cubic=True)
    atoms *= (2, 2, 2)
    atoms.rattle(stdev=0.01, seed=159)

    atoms.info['config_type'] = 'cu_slab'
    atoms.info['buildcell_config_i'] = 'fake_buildecell_config_name'

    return atoms


def test_mult_files(cu_slab, tmp_path):
    ase.io.write(tmp_path / 'f1.xyz', [cu_slab] * 2)
    ase.io.write(tmp_path / 'f2.xyz', cu_slab)
    infiles = [str(tmp_path / 'f1.xyz'), str(tmp_path / 'f2.xyz')]
    inputs = ConfigSet_in(input_files=infiles)
    outputs = ConfigSet_out(output_files={f: f.replace('.xyz', '.out.xyz') for f in infiles})

    calc = EMT()

    atoms_opt = minim.run(inputs, outputs, calc, fmax=1e-2, precon=None,
                          logfile='-', verbose=True, pressure=-1.1215)

    n1 = len(ase.io.read(tmp_path / infiles[0].replace('.xyz', '.out.xyz'), ':'))
    n2 = len(ase.io.read(tmp_path / infiles[1].replace('.xyz', '.out.xyz'), ':'))

    assert n1 == n2 * 2


def test_relax(cu_slab):

    calc = EMT()

    inputs = ConfigSet_in(input_configs = cu_slab)
    outputs = ConfigSet_out()

    atoms_opt = minim.run(inputs, outputs, calc, fmax=1e-2, precon=None,
                          logfile='-', verbose=True, pressure=-1.1215)

    atoms_opt = list(atoms_opt)[-1]

    print('optimised positions:', atoms_opt.positions)

    assert atoms_opt.positions == approx(expected_relaxed_positions_constant_pressure, abs=3e-3)

    assert atoms_opt.info['config_type'] == 'cu_slab_minim_last_converged'


def test_relax_fixed_vol(cu_slab):

    calc = EMT()

    inputs = ConfigSet_in(input_configs = cu_slab)
    outputs = ConfigSet_out()


    atoms_opt = minim.run(inputs, outputs, calc, fmax=1e-2, precon=None,
                          logfile='-', verbose=True)

    atoms_opt = list(atoms_opt)[-1]

    # import json, sys
    # json.dump(atoms_opt.positions.tolist(), sys.stdout)

    expected_positions = np.asarray(
        [[0.0006051702533344901, 0.00045597443454537, 6.327123097568655e-05],
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
         [5.400425465606757, 5.400014020338249, 3.599479942759126]])

    print('optimised positions:', atoms_opt.positions)

    assert atoms_opt.positions == approx(expected_positions, abs=3e-3)

    assert atoms_opt.info['config_type'] == 'cu_slab_minim_last_converged'


def test_subselect_from_traj(cu_slab):

    calc = EMT()

    cu_slab_optimised = cu_slab.copy()
    cu_slab_optimised.set_positions(expected_relaxed_positions_constant_pressure)

    inputs = ConfigSet_in(input_configs = [cu_slab, cu_slab_optimised] )

    # returns full optimisation trajectories
    atoms_opt = minim.run_op(inputs, calc, fmax=1e-2, precon=None,
                          logfile='-', verbose=True, pressure=-1.1215, 
                          steps=2, traj_subselect=None)

    assert len(atoms_opt[0]) == 3
    assert len(atoms_opt[1]) == 1
    assert isinstance(atoms_opt[0][0], Atoms)
    assert isinstance(atoms_opt[1][0], Atoms)

    # returns [None] for unconverged and last config for converged otpimisation
    atoms_opt = minim.run_op(inputs, calc, fmax=1e-2, precon=None,
                          logfile='-', verbose=True, pressure=-1.1215, 
                          steps=2, traj_subselect="last_converged")

    assert len(atoms_opt[1]) == 1
    assert isinstance(atoms_opt[1][0], Atoms) # and not None
    assert atoms_opt[0] is None

    # check that iterable_loop handles Nones as expected
    outputs = ConfigSet_out()
    atoms_opt = minim.run(inputs, outputs, calc, fmax=1e-2, precon=None,
                          logfile='-', verbose=True, pressure=-1.1215, 
                          steps=2, traj_subselect="last_converged")

    atoms_opt = [at for at in atoms_opt]
    assert len(atoms_opt) == 1
    assert isinstance(atoms_opt[0], Atoms)
