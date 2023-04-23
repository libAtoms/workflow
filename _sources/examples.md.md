# Molecular Dynamics

The following script takes atomic structures from "configs.xyz", runs Berendsen NVT molecular dynamics simulation for 6000 steps (3ps) and writes snapshots to "configs.sample.xyz" every 1000 steps (0.5 ps). The script submits jobs to the "standard" queue of "local" cluster on 4 cores each, each job containing 4 MD simulations running in parallel. 

```
import os
from xtb.ase.calculator import XTB
from expyre.resources import Resources
from wfl.autoparallelize import RemoteInfo
from wfl.autoparallelize import AutoparaInfo
from wfl.generate import md
from wfl.configset import ConfigSet, OutputSpec

temp = 300

num_cores=4
steps = 6000
sample_interval = 1000
input_fname = "configs.xyz"
out_fname = "configs.sample.xyz"

num_inputs_per_queued_job = num_cores
max_time = '48h'
partitions="standard"
sysname="local"

# remote info
resources = Resources(
    max_time = max_time,
    num_cores = num_cores,
    partitions = partitions)

remote_info = RemoteInfo(
    sys_name = sysname, 
    job_name = "md", 
    resources = resources, 
    num_inputs_per_queued_job=num_inputs_per_queued_job,
    exact_fit=False, 
    pre_cmds = ["conda activate my-env"]
    ) 

calc = (XTB, [], {'method':'GFN2-xTB'})

ci = ConfigSet(input_fname)
co = OutputSpec(out_fname)

# Needed for the script be re-runable, otherwise a different random seed is generated.
# Without this, if this script is interrupted while the jobs is running, re-starting this 
# script would make it create and submit new jobs rather than monitor the ones already running.
os.environ["WFL_DETERMINISTIC_HACK"] = "true"

# xTB has some internal parallelisation that needs turning off by setting this env. variable. 
os.environ["OMP_NUM_THREADS"] = "1"

ci = md.md(
    inputs=ci, 
    outputs=co,
    calculator=calc,
    steps = steps, 
    dt = 0.5,           
    temperature = temp,  
    temperature_tau = 500, 
    traj_step_interval = sample_interval, 
    results_prefix = "xtb2_", 
    update_config_type = False,
    autopara_info = AutoparaInfo(
        remote_info=remote_info, 
        num_inputs_per_python_subprocess=1)
    )

```

expyre config.json:

```
{ "systems": {
    "local": { "host": null,
        "scheduler": "slurm",
        "commands": ["source ~/.bashrc",  "echo $(date)", "hostname"],
        "header": ["#SBATCH --nodes={nnodes}",
                   "#SBATCH --tasks-per-node={num_cores_per_node}",
                   "#SBATCH --cpus-per-task=1",
                   "#SBATCH --account=change-me",
                   "#SBATCH --qos=standard"],
        "partitions": {"standard" : {"num_cores": 128, "max_time" : "24h", "max_mem": "256G"},
                       "highmem" : {"num_cores": 128, "max_time" : "24h", "max_mem": "512G"}}
                 }
}}

```
