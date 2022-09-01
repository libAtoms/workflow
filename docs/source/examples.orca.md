# ORCA: all parallelisation options 

## 1. Evaluate atomic configurations sequentially on a single core

The following command will evaluate `configs.xyz` with the ORCA DFT code. The Workflow code called with `wfl ref-method orca-eval ... ` is an extension of the [ASE ORCA calculator](https://wiki.fysik.dtu.dk/ase/ase/calculators/orca.html#module-ase.calculators.orca). The extended calculator automates file handling: each structure is evaluated in a separate directory and only specified files are kept. 

```
base_rundir=orca_base_rundir
orca_simple_input="UKS B3LYP def2-SV(P) def2/J D3BJ"
orca_additional_blocks="%scf Convergence Tight SmearTemp 5000 end "
output_prefix='dft_'
orca_command="/opt/womble/orca/orca_5.0.0/orca"
tmp_dir=/scratch-ssd/my_dir_on_scratch
mkdir -p $tmp_dir

input="configs.xyz"
output="configs.dft.xyz"

wfl -v ref-method orca-eval -tmp ${tmp_dir} --base-rundir $base_rundir --output-prefix "${output_prefix}" 
                            --orca-simple-input "${orca_simple_input}" --orca-additional-blocks "${orca_additional_blocks}" 
                            --orca-command "${orca_command}" --output-file $output --keep-files default $input
```

The parameters are 

* `base_rundir` - this directory is where directories for each separate DFT calculation will be based. 
* `orca_simple_input` - [ORCA simple input](https://sites.google.com/site/orcainputlibrary/general-input?authuser=0#h.rafay5vzyzkw) line, prepended with `!`.
* `orca_additional_blocks` - [ORCA block input](https://sites.google.com/site/orcainputlibrary/general-input?authuser=0#h.k0jcjw8bcgcx), the rest of ORCA input file, excluding the coordinate definition.
* `output_prefix` - prefix to the results' keys in `atoms.info` and `atoms.arrays`. Here DFT results will be saved to `dft_energy`, `dft_forces` and `dft_dipole`. 
* `orca_command` - path to ORCA executable.
* `tmp_dir` - directory on a SSD for more efficient calculations. 
* `input` - `.xyz` file to be evaluated. 
* `output` - `.xyz` to save configs with results to. This command will refuse to overwrite if the file exists. 
* `--keep-files default` - only `.inp`, `.out`, `.ase`, `.engrad`, `.xyz` and `_trj.xyz` files will be saved, where applicable, upon success. All files will be kept if the calculation failed.
* `wfl -v ref-method orca-eval` - Workflow command to evaluate DFT, in verbose mode. 

With this command each structure will be evaluated sequentially on a single core. 

## 2.1 Parallelise with ORCA

Changing the simple input to `orca_simple_input="UKS B3LYP def2-SV(P) def2/J D3BJ PAL8"` will turn on [ORCA's parallelsation](https://sites.google.com/site/orcainputlibrary/setting-up-orca#h.p0hdj6lom1lz). This command will still evaluate all structures in `config.xyz` in sequence, but each evaluation will use 8 cores and MPI within ORCA (no need to `mpirun -np 8 orca`)

## 2.2 Parallelise with Workflow

Alternatively, it might be more useful to run ORCA itself sequentially (`orca_simple_input="UKS B3LYP def2-SV(P) def2/J D3BJ"`), but use Workflow to evaluate multiple structures at the same time in parallel. For that, `WFL_NUM_PYTHON_SUBPROCESSES` should be set to the number of processes to use (e.g. `export WFL_NUM_PYTHON_SUBPROCESSES=8`). (N.B. as a rule of thumb, `OMP_NUM_THREADS` must be set to `1`).

## 2.3 Parallelise with ORCA and Workflow

Mixed-mode parallelisation is not supported or tested and should be avoided. We expect that the two cases above should cover practically all of the use-cases. 

## 3. Evaluate structures in multiple queued jobs

The parallelisation steps described above would normally be done as a "queued job" on a cluster. Workflow allows for the input configurations' file to be chopped up and submitted as a separate jobs (possibly using parallelisation within each job) allowing for an even more efficient parallelisation. 

First, one need a configuration (`config.json`) that specifies what "resources" are avaliable on the "local" cluster/machine/queue. For explanation of what means what see ...

```
{ "systems": {
    "local": { "host": null,
        "scheduler": "sge",
        "commands": [  "echo $(date)", "hostname"],
        "header": ["#$ -pe smp {num_cores_per_node}"],
        "rundir": "/data/eg475/run_expyre",
        "partitions": {"any":        {"num_cores": 32, "max_time": "168h", "max_mem": "50GB"},
                       "any@node15": {"num_cores": 16, "max_time": "168h", "max_mem": "47GB"},
                    }
            }
}}
```

For a job to run remotely, one needs to specify a "remote info" dictionary:

```
remote_ifno = {
    "orca.py::evaluate" : {
        "sys_name": "local",
        "job_name": "dft",
        "resources": {"num_cores" : 4, "partitions":"any$", "max_time": "4h"},
        "partial_node":true,
        "num_inputs_per_queued_job": 20}}
```

* `orca.py::evaluate` - the signature of the function which should be parallelised by submitting multiple queued jobs. Scripts might have multiple functions that can be parallelised like this and it might make sense to only do that to some of them.
* `sys_name` - queue/system/cluster name to execute the job on. One fo the entries in `config.json`
* `job_name` - prefix for the job's name on the queue (converted to `#$ -N` bit of the header on SGE).
* `resources` - how many resources the job should require from the queue.
    * `"num_cores": 4` - ask for 4 cores. That's executed by `"header": ["#$ -pe smp {num_cores_per_node}"]` of `config.json`. 
    * `partitions` - name of the queue, converted to `#$ -q any` in this example that uses SGE. ExPyRe uses RegEx to find matching partition names, so `any$` will only pick the first of the two partitions specified in `config.json`. 
    * `max_time` - time requirement for the job, converted to `#$ -l h_rt=4:00:00` in this example.
* `partial_node` - assumes that the que works in "node non-exclusive" way, multiple jobs may be run on a given node and allows picking a partition (`any` in this case) even if it has more cores available (32 here) than the number of cores needed for the job (4 in this case). 
* `num_inputs_per_queued_job` - how many structures to assign to a single queued job. 

## 4. Complete script

Let us assume 100 structures in the `configs.xyz` file. Put together, this script will create 5 separate 4-core jobs (`100/num_inputs_per_queued_job`) with 20 atomic structures assigned to each. Within a given job, the configs will be evaluated 4 at a time (i.e. `WFL_NUM_PYTHON_SUBPROCESSES` will be _automatically_ set to `4`), each ORCA comman running in serial. 

```
export WFL_EXPYRE_INFO=$PWD/remoteinfo.json
cat <<EOF > remoteinfo.json
{
"orca.py::evaluate" : {
    "sys_name": "local",
    "job_name": "dft",
    "resources": {"num_cores" : 4, "partitions":"any$", "max_time": "4h"},
    "partial_node":true,
    "num_inputs_per_queued_job": 20}
}
EOF

mkdir -p _expyre

base_rundir=orca_base_rundir
orca_simple_input="UKS B3LYP def2-SV(P) def2/J D3BJ"
orca_additional_blocks="%scf Convergence Tight SmearTemp 5000 end "
output_prefix='dft_'
orca_command="/opt/womble/orca/orca_5.0.0/orca"
tmp_dir=/scratch-ssd/my_dir_on_scratch
mkdir -p $tmp_dir

input="configs.xyz"
output="configs.dft.xyz"

wfl -v ref-method orca-eval -tmp ${tmp_dir} --base-rundir $base_rundir --output-prefix "${output_prefix}" 
                            --orca-simple-input "${orca_simple_input}" --orca-additional-blocks "${orca_additional_blocks}" 
                            --orca-command "${orca_command}" --output-file $output --keep-files default $input
```

## config.json example

Below is an example of corresponding `config.json`

```
    { "systems": {
        "local": { "host": null,
            "scheduler": "sge",
            "commands": [  "echo $(date)", "hostname"],
            "header": ["#$ -pe smp {num_cores_per_node}"],
            "rundir": "/data/eg475/run_expyre",
            "partitions": {"any":        {"num_cores": 32, "max_time": "168h", "max_mem": "50GB"},
                           "any@node15": {"num_cores": 16, "max_time": "168h", "max_mem": "47GB"},
                        }
                }
    }}
```

## Another complete example

```
from pathlib import Path
from wfl.configset import ConfigSet, OutputSpec
from wfl.calculators.orca import ORCA, natural_population_analysis
import util
from util.util_config import Config
from functools import partial
from wfl.calculators import generic
import wfl.autoparallelize.remoteinfo
from expyre.resources import Resources

input_fname = "validation.rdkit.xtb2_md.dft.both.xyz"
output_fname = "validation.rdkit.xtb2_md.dft.both.dft.xyz"
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

remote_info = wfl.autoparallelize.remoteinfo.RemoteInfo(
    sys_name = "local", 
    job_name = "npa", 
    resources = resources, 
    partial_node = True, 
    num_inputs_per_queued_job=num_inputs_per_queued_job, 
    output_files=["ORCA_calc_files"])

# orca params
orca_kwargs = {"orcablocks": "%scf Convergence Tight SmearTemp 5000 end",
               "orcasimpleinput": "UKS B3LYP def2-SV(P) def2/J D3BJ"}
print(f'orca_kwargs: {orca_kwargs}')

# calculator
keep_files = False 
janpa_home_dir = "/home/eg475/programs/janpa" 
post_func = partial(natural_population_analysis, janpa_home_dir)
calculator = (ORCA, [], {**{"keep_files":keep_files, "post_process":post_func}, **orca_kwargs})

# run calculation
generic.run(
    inputs=ci, 
    outputs=co,
    calculator=calculator,
    properties=["energy", "forces"],
    output_prefix='dft_',
    remote_info=remote_info,
	num_inputs_per_python_subprocess=1)



```



  



