import pytest
import os
import time

import numpy as np

import ase.io
from ase.atoms import Atoms
from ase.calculators.emt import EMT

from wfl.configset import ConfigSet, OutputSpec
from wfl.generate import buildcell
from wfl.calculators import generic
from wfl.autoparallelize import AutoparaInfo

try:
    import torch
    from mace.calculators.foundations_models import mace_mp
except ImportError:
    torch = None


def test_empty_iterator(tmp_path):
    co = buildcell.buildcell(range(0), OutputSpec(tmp_path / 'dummy.xyz'),  buildcell_cmd='dummy', buildcell_input='dummy')

    assert len([at for at in co]) == 0


def test_autopara_info_dict():
    np.random.seed(5)

    ats = []
    nconf = 60
    for _ in range(nconf):
        ats.append(Atoms(['Al'] * nconf, scaled_positions=np.random.uniform(size=(nconf, 3)), cell=[10, 10, 10], pbc=[True] * 3))

    co = generic.calculate(ConfigSet(ats), OutputSpec(), EMT(), output_prefix="_auto_", autopara_info={"num_python_subprocesses": 1})
    assert len(list(co)) == nconf


@pytest.mark.perf
def test_pool_speedup():
    np.random.seed(5)

    rng = np.random.default_rng(5)
    ats = []
    nconf = 60
    at_prim = Atoms('Al', cell=[1, 1, 1], pbc=[True] * 3)
    for _ in range(nconf):
        ats.append(at_prim * (4, 4, 4))
        ats[-1].rattle(rng=rng)

    t0 = time.time()
    co = generic.calculate(ConfigSet(ats), OutputSpec(), EMT(), output_prefix="_auto_",
                           autopara_info=AutoparaInfo(num_python_subprocesses=1,
                                                      num_inputs_per_python_subprocess=30))
    dt_1 = time.time() - t0

    t0 = time.time()
    co = generic.calculate(ConfigSet(ats), OutputSpec(), EMT(), output_prefix="_auto_",
                           autopara_info=AutoparaInfo(num_python_subprocesses=2,
                                                      num_inputs_per_python_subprocess=30))
    dt_2 = time.time() - t0

    print("time ratio", dt_2 / dt_1)
    assert dt_2 / dt_1 < 0.75


@pytest.mark.skipif(torch is None or not torch.cuda.is_available() or os.environ.get("WFL_TORCH_N_GPUS") is None, reason="No torch CUDA devices available, or WFL_TORCH_N_GPUS isn't set")
@pytest.mark.perf
def test_pool_speedup_GPU(monkeypatch):
    np.random.seed(5)

    rng = np.random.default_rng(5)
    ats = []
    nconf = 60
    at_prim = Atoms('Al', cell=[1, 1, 1], pbc=[True] * 3)
    for _ in range(nconf):
        ats.append(at_prim * (5, 5, 5))
        ats[-1].rattle(rng=rng)

    calc = (mace_mp, ["small-omat-0"], {"device": "cuda"})

    req_n_gpus = os.environ["WFL_TORCH_N_GPUS"]
    if len(req_n_gpus) == 0:
        req_n_gpus = str(len(os.environ["CUDA_VISIBLE_DEVICES"].split(",")))

    if "WFL_TORCH_N_GPUS" in os.environ:
        monkeypatch.delenv("WFL_TORCH_N_GPUS")

    t0 = time.time()
    co = generic.calculate(ConfigSet(ats), OutputSpec(), calc, output_prefix="_auto_",
                           autopara_info=AutoparaInfo(num_python_subprocesses=1,
                                                      num_inputs_per_python_subprocess=30))
    dt_1 = time.time() - t0

    monkeypatch.setenv("WFL_TORCH_N_GPUS", req_n_gpus)

    t0 = time.time()
    co = generic.calculate(ConfigSet(ats), OutputSpec(), calc, output_prefix="_auto_",
                           autopara_info=AutoparaInfo(num_python_subprocesses=2,
                                                      num_inputs_per_python_subprocess=30))
    dt_2 = time.time() - t0

    monkeypatch.delenv("WFL_TORCH_N_GPUS")

    print("time ratio", dt_2 / dt_1)
    assert dt_2 / dt_1 < 0.75


def test_outputspec_overwrite(tmp_path):
    with open(tmp_path / "ats.xyz", "w") as fout:
        fout.write("BOB")

    os = OutputSpec("ats.xyz", file_root=tmp_path)
    assert os.all_written()

    ats = []
    nconf = 60
    for _ in range(nconf):
        ats.append(Atoms(['Al'] * nconf, scaled_positions=np.random.uniform(size=(nconf, 3)), cell=[10, 10, 10], pbc=[True] * 3))

    co = generic.calculate(ConfigSet(ats), OutputSpec(), EMT(), output_prefix="_auto_", autopara_info=AutoparaInfo(num_python_subprocesses=1))

    # should skip ops, incorrectly, and ats.xyz doesn't actually contain atoms
    with pytest.raises(ase.io.extxyz.XYZError):
        _ = ase.io.read(tmp_path / "ats.xyz", ":")
