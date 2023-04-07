import numpy as np

from ase.atoms import Atoms

from wfl.configset import ConfigSet, OutputSpec
from wfl.map import map as wfl_map

def displ(at):
    at_copy = at.copy()
    at_copy.positions[:, 0] += 0.1
    return at_copy


def displ_args(at, dx=None, dy=None):
    at_copy = at.copy()
    if dx is not None:
        at_copy.positions += dx
    if dy is not None:
        at_copy.positions += dy

    return at_copy

def test_map(tmp_path):
    inputs = ConfigSet([Atoms('H2', positions=[[0, 0, 0], [2, 0, 0]], cell=[5]*3, pbc=[True]*3) for _ in range(20)])

    displ_inputs = wfl_map(inputs, OutputSpec(tmp_path / "pert_configs.xyz"), displ)
    assert len(list(displ_inputs)) == 20

    displ_inputs_manual = np.asarray([at.positions for at in inputs])
    displ_inputs_manual[:, :, 0] += 0.1

    assert np.allclose(displ_inputs_manual, [at.positions for at in displ_inputs])


def test_map_args(tmp_path):
    inputs = ConfigSet([Atoms('H2', positions=[[0, 0, 0], [2, 0, 0]], cell=[5]*3, pbc=[True]*3) for _ in range(20)])

    # one positional argument
    displ_inputs = wfl_map(inputs, OutputSpec(tmp_path / "pert_configs.xyz"), displ_args, args=[[0.1, 0.0, 0.0]])
    assert len(list(displ_inputs)) == 20

    displ_inputs_manual = np.asarray([at.positions for at in inputs])
    displ_inputs_manual[:, :, 0] += 0.1

    assert np.allclose(displ_inputs_manual, [at.positions for at in displ_inputs])


def test_map_kwargs(tmp_path):
    inputs = ConfigSet([Atoms('H2', positions=[[0, 0, 0], [2, 0, 0]], cell=[5]*3, pbc=[True]*3) for _ in range(20)])

    # one keyword argument
    displ_inputs = wfl_map(inputs, OutputSpec(tmp_path / "pert_configs.xyz"), displ_args, kwargs={"dx": [0.1, 0.0, 0.0]})
    assert len(list(displ_inputs)) == 20

    displ_inputs_manual = np.asarray([at.positions for at in inputs])
    displ_inputs_manual[:, :, 0] += 0.1

    assert np.allclose(displ_inputs_manual, [at.positions for at in displ_inputs])

def test_map_args_kwargs(tmp_path):
    inputs = ConfigSet([Atoms('H2', positions=[[0, 0, 0], [2, 0, 0]], cell=[5]*3, pbc=[True]*3) for _ in range(20)])

    # one keyword argument
    displ_inputs = wfl_map(inputs, OutputSpec(tmp_path / "pert_configs.xyz"), displ_args, args=[ [0.05, 0.0, 0.0] ], kwargs={"dy": [0.05, 0.0, 0.0]})
    assert len(list(displ_inputs)) == 20

    displ_inputs_manual = np.asarray([at.positions for at in inputs])
    displ_inputs_manual[:, :, 0] += 0.1

    assert np.allclose(displ_inputs_manual, [at.positions for at in displ_inputs])
