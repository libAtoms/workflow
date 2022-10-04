# ORCA via command line 

## 1. Evaluate atomic configurations sequentially on a single core

The following command will evaluate `configs.xyz` with the ORCA DFT code. The Workflow code called with `wfl ref-method orca-eval ... ` is an extension of the [ASE ORCA calculator](https://wiki.fysik.dtu.dk/ase/ase/calculators/orca.html#module-ase.calculators.orca). The extended calculator automates file handling: each structure is evaluated in a separate directory and only specified files are kept. 

```
scratchdir=/scratch-ssd/my_dir_on_scratch
mkdir -p $scratchdir

wfl -v ref-method orca-eval \
    --scratchdir $scratchdir \
    --workdir orca_workdir \
    --output-prefix dft_ \
    --orcasimpleinput "UKS B3LYP def2-SV(P) def2/J D3BJ" \
    --orcablocks "%scf Convergence Tight SmearTemp 5000 end" \
    --calculator-exec "path/to/orca/orca_5.0.0/orca" \
    --output-file configs.dft.xyz \
    --keep-files default \
    configs.xyz 
```

The parameters are 

* `workdir` - this directory is where directories for each separate DFT calculation will be based. 
* `orcasimpleinput` - [ORCA simple input](https://sites.google.com/site/orcainputlibrary/general-input?authuser=0#h.rafay5vzyzkw) line, during the calculation automatically prepended with `!`.
* `orcablocks` - [ORCA block input](https://sites.google.com/site/orcainputlibrary/general-input?authuser=0#h.k0jcjw8bcgcx), the rest of ORCA input file, excluding the coordinate definition.
* `output_prefix` - prefix to the results' keys in `atoms.info` and `atoms.arrays`. Here DFT results will be saved to `dft_energy`, `dft_forces` and `dft_dipole`. 
* `calculator-exec` - path to ORCA executable.
* `scratchdir` - directory on a SSD for more efficient calculations. 
* `input` - `.xyz` file to be evaluated. 
* `output` - `.xyz` to save configs with results to. This command will refuse to overwrite if the file exists. 
* `--keep-files default` - only `.inp`, `.out`, `.ase`, `.engrad`, `.xyz` and `_trj.xyz` files will be saved, where applicable, upon success. All files will be kept if the calculation failed.
* `wfl -v ref-method orca-eval` - Workflow command to evaluate DFT, in verbose mode. 

With this command each structure will be evaluated sequentially on a single core. 


## 2.1 Parallelize with ORCA

Adding `PAL8` to the simple input (`orcasimpleinput="UKS B3LYP def2-SV(P) def2/J D3BJ PAL8"`) will turn on [ORCA's parallelsation](https://sites.google.com/site/orcainputlibrary/setting-up-orca#h.p0hdj6lom1lz). This command will still evaluate all structures in `config.xyz` in sequence, but each evaluation will use 8 cores and MPI within ORCA (no need to `mpirun -np 8 orca`)


## 2.2 Parallelize with Workflow

Alternatively, it might be more useful to run ORCA itself sequentially (`orcasimpleinput="UKS B3LYP def2-SV(P) def2/J D3BJ"`), but use Workflow to evaluate multiple structures at the same time in parallel. For that, `WFL_NUM_PYTHON_SUBPROCESSES` should be set to the number of processes to use (e.g. `export WFL_NUM_PYTHON_SUBPROCESSES=8`). (N.B. as a rule of thumb, `OMP_NUM_THREADS` must be set to `1`).


## 2.3 Parallelize with ORCA and Workflow

Mixed-mode parallelization is not tested and not supported.


## 3. Evaluate structures in multiple queued jobs from command line

The parallelization steps described above would normally be submitted as a job to a cluster queue. Workflow allows for the input configurations' file to be chopped up and submitted as a separate jobs (most commonly using parallelization within each job) allowing for an even more efficient parallelization. 

First, one need a configuration (`config.json`) that specifies what "resources" are available on the "local" cluster. See [First Example](first_example.md) and [Expyre documentation](https://libatoms.github.io/ExPyRe/) for more details. 

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

For a job to run remotely, one needs to specify a "remote info" dictionary. In this example, the dictionary is stored in a `remoteinfo.json` file. Setting `WFL_EXPYRE_INFO` to path to `remoteinfo.json` will automatically turn on the job submission and parallelization. 

```
remote_info = {
    "generic.py::run" : {
        "sys_name": "local",
        "job_name": "dft",
        "resources": {"num_cores" : 4, "partitions":"any$", "max_time": "4h"},
        "partial_node":true,
        "num_inputs_per_queued_job": 20,
        "pre_cmds": ["conda activate my_env"]}}
```

* `generic.py::run` - the signature of the function which should be parallelized by submitting multiple queued jobs. Scripts might have multiple functions that can be parallelized like this and it might make sense to only do that to some of them.
* `sys_name` - cluster name to execute the job on. One fo the entries in `config.json`
* `job_name` - prefix for the job's name on the queue (converted to `#$ -N` bit of the header on SGE).
* `resources` - what resources the job should require from the queue.
    * `"num_cores": 4` - ask for 4 cores. That's executed by `"header": ["#$ -pe smp {num_cores_per_node}"]` of `config.json`. 
    * `partitions` - name of the queue, converted to `#$ -q any` in this example that uses SGE. ExPyRe uses RegEx to find matching partition names, so `any$` will only pick the first of the two partitions specified in `config.json`. 
    * `max_time` - time requirement for the job, converted to `#$ -l h_rt=4:00:00` in this example.
* `partial_node` - assumes that the queue works in "node non-exclusive" way, multiple jobs may be run on a given node and allows picking a partition (`any` in this case) even if it has more cores available (32 here) than the number of cores needed for the job (4 in this case). 
* `num_inputs_per_queued_job` - how many structures to assign to a single queued job. 

### Complete script

Let us assume 100 structures in the `configs.xyz` file. Put together, this script will create 5 separate 4-core jobs (`100/num_inputs_per_queued_job`) with 20 atomic structures assigned to each. Within a given job, the configs will be evaluated 4 at a time (i.e. `WFL_NUM_PYTHON_SUBPROCESSES` will be _automatically_ set to `4`), each ORCA command running in serial. NB `.expyre/config.json` must be present. 

```
export WFL_EXPYRE_INFO=$PWD/remoteinfo.json
cat <<EOF > remoteinfo.json
{
"generic.py::run" : {
    "sys_name": "local",
    "job_name": "dft",
    "resources": {"num_cores" : 4, "partitions":"any$", "max_time": "4h"},
    "partial_node":true,
    "num_inputs_per_queued_job": 20,
    "pre_cmds": ["conda activate my_env"]}
}
EOF

mkdir -p _expyre

scratchdir=/scratch-ssd/my_dir_on_scratch
mkdir -p $scratchdir

wfl -v ref-method orca-eval \
    --scratchdir $scratchdir \
    --workdir orca_workdir \
    --output-prefix dft_ \
    --orcasimpleinput "UKS B3LYP def2-SV(P) def2/J D3BJ" \
    --orcablocks "%scf Convergence Tight SmearTemp 5000 end" \
    --calculator-exec "path/to/orca/orca_5.0.0/orca" \
    --output-file configs.dft.xyz \
    --keep-files default \
    configs.xyz 
```



  



