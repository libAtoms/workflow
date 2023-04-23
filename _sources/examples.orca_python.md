# ORCA via Python script

This example Python script evaluates atomic structures stored in a "configs.xyz" file with ORCA calculator and saves them to "configs.dft.xyz". The script (`python orca_eval.py` called from head node) submits (multiple) jobs to the cluster, each job containing up to 20 structures (`num_inputs_per_queued_job = 20`) and running on 4 cores (`num_cores = 4`). Once the jobs are submitted, the python execution may be stopped. To save the evaluated structures from all of the finished jobs, the (unchanged!!) script is run again - it checks that the jobs are successfully completed and collects all of their outputs to "configs.dft.xyz".  

Workflow's ORCA calculator has an option to execute a (partially initialized) function just after the ORCA executable has been called. This allows to further process outputs of the calculation before the no longer needed files are deleted. The example below uses external [JANPA](https://sourceforge.net/p/janpa/wiki/Home/) package to take the large ORCA wavefunction files, calculate localized charges and electron populations and save them to `Atoms.arrays` in the output file. 


`orca_eval.py`:
```
from pathlib import Path
from wfl.configset import ConfigSet, OutputSpec
from wfl.calculators.orca import ORCA, natural_population_analysis
from functools import partial
from wfl.calculators import generic
from wfl.autoparallelize import RemoteInfo
from wfl.autoparallelize import AutoparaInfo
from expyre.resources import Resources

input_fname = "configs.xyz"
output_fname = "configs.dft.xyz"
num_inputs_per_queued_job = 20
num_cores = 4
max_time = "48h"

# structures
ci = ConfigSet(input_fname)
co = OutputSpec(output_fname)

expyre_dir = Path("_expyre")
expyre_dir.mkdir(exist_ok=True)

# remote info
resources = Resources(
    max_time = max_time,
    num_cores = num_cores,
    partitions = "any$")

remote_info = RemoteInfo(
    sys_name = "local", 
    job_name = "npa", 
    resources = resources, 
    partial_node = True, 
    num_inputs_per_queued_job=num_inputs_per_queued_job, 
    pre_cmds=["conda activate my_env"])

# orca params
orca_kwargs = {"orcablocks": "%scf Convergence Tight SmearTemp 5000 end",
               "orcasimpleinput": "UKS B3LYP def2-SV(P) def2/J D3BJ", 
               "keep_files": False}
print(f'orca_kwargs: {orca_kwargs}')

# optionally, can define a post-processing function. 
# here uses JANPA to calculate local charges from wavefunction.
janpa_home_dir = "/path/to/janpa/dir" 
post_func = partial(natural_population_analysis, janpa_home_dir)
orca_kwargs["post_process"] = post_func

# calculator
calculator = (ORCA, [], orca_kwargs)

# run calculation
generic.calculate(
    inputs=ci, 
    outputs=co,
    calculator=calculator,
    properties=["energy", "forces"],
    output_prefix='dft_',
    autopara_info = AutoparaInfo(
        remote_info=remote_info,
        num_inputs_per_python_subprocess=1))

```

An example of `~/.expyre/config.json` that stores information about the resources available on the `local` cluster:

```
{ "systems": {
    "local": { "host": null,
        "scheduler": "sge",
        "commands": [  "echo $(date)", "hostname"],
        "header": ["#$ -pe smp {num_cores_per_node}"],
        "partitions": {"any":        {"num_cores": 32, "max_time": "168h", "max_mem": "50GB"},
                       "any@node15": {"num_cores": 16, "max_time": "168h", "max_mem": "47GB"}
                    }
            }
}}
```



  



