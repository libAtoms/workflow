(first_example)=
# First Example

One of the main uses of Workflow is to parallelise per-`Atoms` operations over multiple cores. 

For example, let's say we have created a number of atomic structures ("configurations" or "configs"):

```
from ase.build import bulk
atoms = []
for idx in range(320):
    at = bulk("Cu", "fcc", a=3.6, cubic=True)
    at *= (2, 2, 2)
    at.rattle(stdev=0.01, seed=159+idx)
    atoms.append(at)
```

One of the most common type of operations is to evaluate energies, forces and stresses of these structures, here with an [EMT calculator](https://wiki.fysik.dtu.dk/ase/ase/calculators/emt.html#module-ase.calculators.emt) included in ASE. Normally, the these properties would be obtained and saved by looping through the structures:

```
from ase.calculators.emt import EMT
calc = EMT()

for at in atoms:
    at.calc = EMT()
    at.info["emt_energy"] = at.get_potential_energy()
    at.arrays["emt_forces"] = at.get_forces()
    at.info["emt_stress"] = at.get_stress()

write("configs.emt.xyz", atoms)
``` 

Often property evaluation for each structure might take minutes or hours and there might be hundreds or thousands of configs in the dataset. In such cases the evaluation is performed as a job on a supercomputer and Workflow may be used to efficiently parallelise the property evaluation over all cores available to the job. For this, the script above is writen as

```
from ase.calculators.emt import EMT 
from wfl.calculators import generic
from wfl.autoparallelize import AutoparaInfo
from wfl.configset import ConfigSet, OutputSpec

configset = ConfigSet(atoms)
outputspec = OutputSpec("configs.emt.xyz")
calculator = (EMT, [], {})

generic.calculate(
    inputs = configset, 
    outputs = outputspec, 
    calculator = calculator, 
    output_prefix = "emt_", 
    autopara_info = AutoparaInfo(
        num_python_subprocesses = 8    
    )
)
```

- `generic.calculate()` is the main function that is used to parallelise any* ASE calculator. It returns a ConfigSet object with configs containing the results. 
- `ConfigSet` and `OutputSpec` are used to specify where to read the atomic configs from and write the processed configs to. There are a number of ways to specify read/write destination (e.g. `list(Atoms)` or file(s) as above) when creating `ConfigSet` and `OutputSpec`, but functions such as `generic.calculate()` can have a consistent behaviour irrespective of where the configs came from. 
- `calculator` is specified in a way that is pickleable and may be used in spawned processes of `multiprocessing` package that is used by Workflow. It should be given as a tuple of `(calculator_constructor_function, arguments, keyword_arguments)`. For example, if a quippy calculator is normally instantiated as 

    ```
    from quippy.potential import Potential
    calculator = Potential("TB DFTB", param_filename="path/to/params.xml
    ```

    then for `generic.calculate()` it should be constructed as 

    ```
    from quippy.potential import Potential`
    calculator = (Potential, ["TB DFTB"], {"param_filename":"path/to/params.xml"})
    ```

- `output_prefix` is prepended to each of the calculated properties, saved in `Atoms.info` or `Atoms.arrays` as appropriate. 
- `autopara_info` is used to control the parallelisation. `num_python_subprocesses` specifies how many parallel processes to spawn; a common option is to set as many as cores available for the job. More conveniently, `num_python_subprocesses` is set via a `WFL_NUM_PYTHON_SUBPROCESSES` environment variable. 

## Complete script

`evaluate_emt.py`
```

from ase.build import bulk
from ase.calculators.emt import EMT 
from wfl.calculators import generic
from wfl.autoparallelize import AutoparaInfo
from wfl.configset import ConfigSet, OutputSpec
from wfl.autoparallelize.remote info import RemoteInfo()
from expyre.resources import Resources

atoms = []
for idx in range(32):
    at = bulk("Cu", "fcc", a=3.6, cubic=True)
    at *= (2, 2, 2)
    at.rattle(stdev=0.01, seed=159+idx)
    atoms.append(at)

configset = ConfigSet(atoms)
outputspec = OutputSpec("configs.emt.xyz")
calculator = (EMT, [], {})

generic.calculate(
    inputs = configset, 
    outputs = outputspec, 
    calculator = calculator, 
    output_prefix = "emt_")
```

`sun_grid_engine_sub.sh`
```
#!/bin/bash
#$ -pe smp 16        # number of cores requested
#$ -l h_rt=48:00:00  # time requested in HH:MM:SS format
#$ -S /bin/bash      # shell to run the job in
#$ -N eval_emt       # job name 
#$ -j yes            # combine error and output logs
#$ -cwd              # execute job in directory from which it was submitted
#$ -q 'standard'

export WFL_NUM_PYTHON_SUBPROCESSES=$NSLOTS
python evaluate_emt.py
```


# With remote execution

Workflow also allows to submit (remotely) queued jobs automatically, by interfacing with ExPyRe ([docummentation](https://libatoms.github.io/ExPyRe/), [repository](https://github.com/libAtoms/ExPyRe/tree/main/expyre)). In this example, instead of calling the above python script in a queue submission script, the modified python script is called from the head node and the parallelisation mechanism behind `generic.calculate()` sets up and submits the job and returns the results like the script normally would. To enable remote submission, `RemoteInfo` must be added to `AutoparaInfo`. 

```
from wfl.autoparallelize import RemoteInfo()
from expyre.resources import Resources

remote_info = RemoteInfo(
    sys_name = "local",
    job_name = "eval_emt",
    num_inputs_per_queued_job = 160,
    resources = Resources(
        max_time = "48h",
        num_nodes = 1,
        partitions = "standard"))
```

- `sys_name` picks which cluster to submit the job to. In this example, the script is executed from head node and jobs are submitted to compute nodes on the same cluster. Another example could execute the script on a laptop and submit the job over ssh to one of a number of accessible clusters. 
- `job_name` sets a prefix to each of the submitted jobs. 
- `num_inputs_per_queued_job` specifies how many atomic structures (in this example) to assign to each job. Since the total number is 320, setting this argument to 160 means that in total two jobs will be submitted to the queue. 
- `resources` sets which partition(s) (queues) the job will be sent to and what computational resources will be required in the automatically generated submission script.

The available clusters are listed in `config.json` file, by default at `~/.expyre/config.json`:

```
"local": { "host": null,
        "scheduler": "sge",
        "commands": ["conda activate myenv"],
        "header": ["#$ -pe smp {num_cores}"],
        "partitions": {"standard" : {"num_cores": 16, "max_time": "168h", "max_mem": "200GB"}}
             }
}}
```

- `host` is set to `null`, because this example script is run on the head node of the cluster to which jobs are submitted. I.e. no submission via ssh is needed. 
- `scheduller` - the example cluster uses Sun Grid Engine to manage the queued jobs
- `commands` gives list of commands to execute in the automatically created submission script before a copy of the autoparallelised operation is launched. 
- `header` - any extra header lines to add.  
- `partitions` - names and resources available for the partitions (queues) of the cluster. 


## Complete script

`evaluate_emt_remotely.py`
```

from ase.build import bulk
from ase.calculators.emt import EMT 
from wfl.calculators import generic
from wfl.autoparallelize import AutoparaInfo
from wfl.configset import ConfigSet, OutputSpec
from wfl.autoparallelize import RemoteInfo
from expyre.resources import Resources

atoms = []
for idx in range(32):
    at = bulk("Cu", "fcc", a=3.6, cubic=True)
    at *= (2, 2, 2)
    at.rattle(stdev=0.01, seed=159+idx)
    atoms.append(at)

configset = ConfigSet(atoms)
outputspec = OutputSpec("configs.emt.xyz")
calculator = (EMT, [], {})

remote_info = RemoteInfo(
    sys_name = "local",
    job_name = "eval_emt",
    num_inputs_per_queued_job = 16,
    resources = Resources(
        max_time = "48h",
        num_nodes = 1,
        partitions = "standard"))

generic.calculate(
    inputs = configset, 
    outputs = outputspec, 
    calculator = calculator, 
    output_prefix = "emt_", 
    autopara_info = AutoparaInfo(
        remote_info = remote_info))
```
