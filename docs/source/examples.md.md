# MD

takes input file, and runs many md trajectories and saves each to a different trajectory file. 
(single-to-many configset.py functionality would be very useful here)

Script to submit jobs

```
from ase.io import read
from pathlib import Path
import util.md.test
from expyre.resources import Resources
import wfl.pipeline.utils


temp = 300
nnodes=10
steps = 300000
sample_interval = 5000
ace_fname="/work/e89/e89/eg475/work/md_tests/ace_cut4.5_N3D16_ard/2_wdir/ace_cut4.5_ard_N3D16.json"
input_fname = "mega_md_test_in.xyz"
wdir = "md_outputs"

num_inputs_per_queued_job = nnodes * 128
max_time = '24h'
n = (nnodes, "nodes")
partitions="standard"
sysname="local"
partial_node=False

Path("_expyre").mkdir(exist_ok=True)

# remote info
resources = Resources(
    max_time = max_time,
    n = n,
    partitions = partitions)

remote_info = wfl.pipeline.utils.RemoteInfo(
    sys_name = sysname,
    job_name = "md-test",
    resources = resources,
    partial_node = partial_node,
    num_inputs_per_queued_job=num_inputs_per_queued_job,
    )


calc = (pyjulip_ace, [ace_fname], {})
ats = read(input_fname, ":")
wdir = Path(wdir) / str(int(temp))
wdir.mkdir(exist_ok=True, parents=True)

run_md_test(
    workdir_root=wdir,
    in_ats=ats,
    temp=temp,
    calc=calc,
    info_label="graph_name",
    steps=steps,
    sampling_interval=sample_interval,
    pred_prop_prefix="ace_" ,
    remote_info=remote_info
    )

```

Extra functions:

```
import os
from pathlib import Path
from wfl.generate_configs import md
import numpy as np
import hashlib

def pyjulip_ace(ace_fname):
    import pyjulip
    return pyjulip.ACE1(ace_fname)


def run_md_test(workdir_root, in_ats, temp, calc, info_label, steps, sampling_interval, 
        pred_prop_prefix, remote_info):

    workdir_root = Path(workdir_root) 
    workdir_root.mkdir(exist_ok=True)

    ci, co = prepare_inputs(in_ats, info_label, workdir_root)

    os.environ["WFL_DETERMINISTIC_HACK"] = "true"

    md_params = {
        "steps": steps,
        "dt": 0.5,  # fs
        "temperature": temp,  # K
        "temperature_tau": 500,  # fs, somewhat quicker than recommended (???)
        "traj_step_interval": sampling_interval,
        "results_prefix": pred_prop_prefix}

    md.sample(
        inputs=ci, 
        outputs=co,
        calculator=calc,
        verbose=False,
        remote_info=remote_info,
        **md_params
        )

def prepare_inputs(ats, info_label, workdir_root):

    input_files = []
    output_files = {}
    for at in ats:
        hash = hash_atoms(at)
        label = at.info[info_label] + hash
        fname_in = workdir_root / (label + "_in.xyz")
        fname_out = workdir_root / (label + "_out.xyz")
        input_files.append(fname_in)
        output_files[fname_in] = fname_out
        write(fname_in, at)

    ci = ConfigSet(input_files)
    co = OutputSpec(output_files)
    return ci, co

#creates unique hash for a matrix of numbers
def hash_array(v):
    return hashlib.md5(np.array2string(v, precision=8, sign='+', floatmode='fixed').encode()).hexdigest()

#creates unique hash for Atoms from atomic numbers and positions
def hash_atoms(at):
    v = np.concatenate((at.numbers.reshape(-1,1), at.positions),axis=1)
    return hash_array(v)

```

expyre config.json:

```
{ "systems": {
    "local": { "host": null,
        "scheduler": "slurm",
        "commands": ["source /work/e89/e89/eg475/.bashrc",  "echo $(date)", "hostname"],
        "header": ["#SBATCH --nodes={nnodes}",
                   "#SBATCH --tasks-per-node={num_cores_per_node}",
                   "#SBATCH --cpus-per-task=1",
                   "#SBATCH --account=change-me",
                   "#SBATCH --qos=standard"],
                "rundir": "/work/e89/e89/eg475/expyre_rundir",
        "partitions": {"standard" : {"num_cores": 128, "max_time" : "24h", "max_mem": "256G"},
                       "highmem" : {"num_cores": 128, "max_time" : "24h", "max_mem": "512G"}}
                 }
}}

```
