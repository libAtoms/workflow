# check that rng for an op after one that's skipped is the same as it was when
# the initial op actually did some work

import numpy as np

from ase.calculators.emt import EMT
from ase.atoms import Atoms

from wfl.configset import ConfigSet, OutputSpec
from wfl.generate.md import md

ats = ConfigSet([Atoms('Al' * 8, positions=[[0, 0, 0],
                                            [2, 0, 0],
                                            [0, 2, 0],
                                            [2, 2, 0],
                                            [0, 0, 2],
                                            [2, 0, 2],
                                            [0, 2, 2],
                                            [2, 2, 2]], cell=[4] * 3, pbc=[True] * 3) for _ in range(12)])

def test_rng_determinism(tmp_path):
    ####################################################################################################
    # do two MDs, one after the other
    print("two MDs, one after the other")

    rng = np.random.default_rng(1)

    # NVE with random initial velocities
    ats_md = md(ats, OutputSpec("TEST_0_initial_md_out.extxyz", file_root=tmp_path), calculator=(EMT, [], {}),
                dt=1.0, steps=100, traj_step_interval=50,
                temperature=300, rng=rng, autopara_info={'num_inputs_per_python_subprocess': 3})
    vs_0_0 = np.asarray([at.get_velocities()[0, 0] for at in ats_md])
    print("initial MD", vs_0_0)

    # NVE with random initial velocities
    ats_md = md(ats, OutputSpec("TEST_0_second_md_out.extxyz", file_root=tmp_path), calculator=(EMT, [], {}),
                dt=1.0, steps=100, traj_step_interval=50,
                temperature=300, rng=rng, autopara_info={'num_inputs_per_python_subprocess': 3})
    vs_0_1 = np.asarray([at.get_velocities()[0, 0] for at in ats_md])
    print("second MD", vs_0_1)

    # make sure 2nd run is different from first, i.e. rng changed state
    assert not np.all(vs_0_0 == vs_0_1)

    print("")
    ####################################################################################################
    # fake initial MD, then do the second

    print("do an MD, then repeat it and do a second")
    rng = np.random.default_rng(1)
    with open(tmp_path / "TEST_1_initial_md_out.extxyz", "w") as fout:
        fout.write("\n")

    # NVE with random initial velocities
    ats_md = md(ats, OutputSpec("TEST_1_initial_md_out.extxyz", file_root=tmp_path), calculator=(EMT, [], {}),
                dt=1.0, steps=100, traj_step_interval=50,
                temperature=300, rng=rng, autopara_info={'num_inputs_per_python_subprocess': 3})
    vs_1_0 = np.asarray([at.get_velocities()[0, 0] for at in ats_md])
    print("fake initial MD", vs_1_0)

    # NVE with random initial velocities
    ats_md = md(ats, OutputSpec("TEST_1_second_md_out.extxyz", file_root=tmp_path), calculator=(EMT, [], {}),
                dt=1.0, steps=100, traj_step_interval=50,
                temperature=300, rng=rng, autopara_info={'num_inputs_per_python_subprocess': 3})
    vs_1_1 = np.asarray([at.get_velocities()[0, 0] for at in ats_md])
    print("second MD", vs_1_1)

    # make sure that vs from second run after a skipped first run are the same as they were before,
    # when first run actually did something
    assert np.all(vs_0_1 == vs_1_1)

    print("")
    ####################################################################################################
    # fake initial MD, then do the second

    print("do an MD, then repeat it and do a second")
    rng = np.random.default_rng(1)
    with open(tmp_path / "TEST_2_initial_md_out.extxyz", "w") as fout:
        fout.write("\n")

    # NVE with random initial velocities
    ats_md = md(ats, OutputSpec("TEST_2_initial_md_out.extxyz", file_root=tmp_path), calculator=(EMT, [], {}),
                dt=1.0, steps=100, traj_step_interval=50,
                temperature=300, rng=rng, autopara_info={'num_inputs_per_python_subprocess': 3})
    vs_2_0 = np.asarray([at.get_velocities()[0, 0] for at in ats_md])
    print("fake initial MD", vs_2_0)

    # NVE with random initial velocities
    ats_md = md(ats, OutputSpec("TEST_2_second_md_out.extxyz", file_root=tmp_path), calculator=(EMT, [], {}),
                dt=1.0, steps=100, traj_step_interval=50,
                temperature=300, rng=rng, autopara_info={'num_inputs_per_python_subprocess': 4})
    vs_2_1 = np.asarray([at.get_velocities()[0, 0] for at in ats_md])
    print("second MD with different inputs per python subprocess", vs_2_1)

    # make sure that vs from second run after a skipped first run are the same as they were before,
    # when first run actually did something, even if split into python subprocesses is different
    assert  np.all(vs_0_1 == vs_2_1)
