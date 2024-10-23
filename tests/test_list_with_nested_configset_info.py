from wfl.configset import ConfigSet, OutputSpec
from wfl.calculators.generic import calculate
from wfl.generate.md import md

import ase.io
from ase.atoms import Atoms
from ase.calculators.emt import EMT

import pytest

pytestmark = pytest.mark.remote

def test_list_with_nested_configset_info(tmp_path, expyre_systems, remoteinfo_env):
    for sys_name in expyre_systems:
        if sys_name.startswith('_'):
            continue

        do_test_list_with_nested_configset_info(tmp_path, sys_name, remoteinfo_env)

def do_test_list_with_nested_configset_info(tmp_path, sys_name, remoteinfo_env):
    ri = {'sys_name': sys_name, 'job_name': 'pytest_'+sys_name,
          'resources': {'max_time': '1h', 'num_nodes': 1},
          'num_inputs_per_queued_job': 1, 'check_interval': 10}

    remoteinfo_env(ri)

    print('RemoteInfo', ri)

    cs = [Atoms('Al', cell=[3.0] * 3, pbc=[True] * 3) for _ in range(20)]

    for at_i, at in enumerate(cs):
        at.info["_ConfigSet_loc"] = f" / {at_i} / 0 / 10000"

    ase.io.write(tmp_path / "tt.extxyz", cs)

    configs = ConfigSet("tt.extxyz", file_root=tmp_path)

    os = OutputSpec("t1.extxyz", file_root=tmp_path)
    cc = md(configs, os, (EMT, [], {}), steps=10, dt=1.0, autopara_info={'remote_info': ri})

    traj_final_configs = []
    for traj_grp in cc.groups():
        traj_final_configs.append([atoms for atoms in traj_grp][-1])

    # ConfigSet should work
    print("trying t2")
    configs = ConfigSet(traj_final_configs)

    os = OutputSpec("t2.extxyz", file_root=tmp_path)
    _ = md(configs, os, (EMT, [], {}), steps=10, dt=1.0, autopara_info={'remote_info': ri})

    # list originally failed, as in https://github.com/libAtoms/workflow/issues/344
    print("trying t3")
    configs = traj_final_configs

    os = OutputSpec("t3.extxyz", file_root=tmp_path)
    _ = md(configs, os, (EMT, [], {}), steps=10, dt=1.0, autopara_info={'remote_info': ri})
