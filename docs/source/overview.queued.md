
# Functions as independently queued jobs

Operations that have been wrapped in `autoparallelize` can be split into independent jobs with minimal
coding.  A `config.json` file must describe the available queuing sytem resources (see
[expyre README](https://github.com/libAtoms/ExPyRe#readme)), and a JSON file (content or path in
env var `WFL_EXPYRE_INFO`) describes the resources needed by any `autoparallelize` call that
should be executed this way.  Any remote machine to be used requires that the `wfl` python
module be installed.  If needed, commands needed to make this module available (e.g. setting `PYTHONPATH`)
can be set on a per-machine basis in the `config.json` file mentioned below.

```{warning}
To facilitate restarts of interrupted operations, submitted jobs are cached.  If the code 
executed by the job is changed, this may result in cached but incorrect output being used.
See [discussion below](sec:example:restarts).
```

In addition, `wfl.fit.gap_simple`, `wfl.fit.gap_multistage`, and `wfl.fit.ace` have been wrapped, as a single
job each.  The GAP simple fit is controlled by the `WFL_GAP_SIMPLE_FIT_REMOTEINFO` env var.  Setting
this variable will also lead to the multistage fit submitting each simple fit as its own job.
In addition, the multistage fit can be turned into a single job containing all the stages
with the `WFL_GAP_MULTISTAGE_FIT_REMOTEINFO` env var.  In principle, doing each simple fit
as its own job could enable running committee fits in parallel, but that is not supported right now.
The env var `WFL_ACE_FIT_REMOTEINFO` is used for ACE fits.

```{note}
Now that the multistage fit does very little other than the repeated simple fitting, does
it need its own level of remote job execution?
```

The `*REMOTEINFO` and `WFL_EXPYRE_INFO` environment variables allow to flexibly control which parts of 
a (likely long and multi-file) fitting script are executed remotely and with what resources without a need
to change the script itself thus allowing for more flexibility. For simpler scripts, `RemoteInfo` python object
may be given to the to-be remotely submitted function instead of setting the environment variables. 


(sec:example)=
## Example

The workflow (`do_workflow.py`) is essentially identical to what you'd otherwise construct:
```
from wfl.configset import ConfigSet, OutputSpec
from wfl.generate_configs.minim import run

from ase.calculators.vasp import Vasp

infile = "structures.xyz"
ci = ConfigSet(infile)
co = OutputSpec('relaxed.' + infile)

vasp_kwargs = { 'encut': 400.0, 'ismear': 0, 'sigma': 0.1, 'ediff': 1.0d-7, 'kspacing': 0.15, 'kgamma': True }

# set path to POTCARs
# override non-trivial default setups, just use file in POTCARs/<symbol>/POTCAR
vasp_kwargs['pp'] = '.' # override ASE Vasp's unneeded "potpaw*" addition
vasp_kwargs['setups'] = { chemsym: '' for chemsym in ase.data.atomic_numbers.keys() }

#   do relaxation
ci_relaxed = minim_run(ci, co, calculator=(Vasp, [], vasp_kwargs), pressure=0.0)
```

The interactive commands to prepare for running the workflow set up the env vars necessary to control the queued jobs:
```
export WFL_EXPYRE_INFO=$PWD/remoteinfo.json
cat<<EOF > remoteinfo.json
{
 "minim.py::run" : {
     "sys_name": "${sys_name}",
     "job_name": "vasp_minim",
     "resources": { "num_nodes" : 1, "max_time": "24h" },
     "num_inputs_per_queued_job" : 1,
     "input_files": ["POTCARs"],
     "env_vars": ["VASP_COMMAND=${vasp_path}", "VASP_PP_PATH=POTCARs",
                  "WFL_NUM_PYTHON_SUBPROCESSES=\${EXPYRE_NCORES_PER_NODE}",
                  "WFL_VASP_KWARGS='{ \"ncore\": '\${EXPYRE_NCORES_PER_NODE}'}'" ]
   }
}
EOF

python3 ./do_workflow.py
```
where `${sys_name}` is set to the name of the system you want to run on, and `${vasp_path}$`
is the path to the vasp executable on the remote machine.

In addition a `config.json` file (see example in [expyre README](https://github.com/libAtoms/ExPyRe#readme))
with available systems needs to be created
(usually in `$HOME/.expyre/`, otherwise in the `_expyre` directory
mentioned below), and a directory called `some/path/_expyre/` (note
the initial `_`, not `.`, so it is more visible) can optionally be created at
the directory hierarchy level that indicates the scope of the project,
to separate the jobs database from any other project.

(sec:example:restarts)=
### Restarts

Restarts are supposed to be handled automatically - if the workflow script is
interrupted, just rerun it.  If the entire `autoparallelize` call is complete,
the default behavior of `OutputSpec` will allow
it to skip the operation entirely.  If the operation is not entirely done,
the remote running package will detect an attempt to compute a previously
initiated call (based on a hash of pickles of the function and all of its
arguments) and not duplicate the job submission.  Note that this hashing
mechanism is not perfect, and functions that pass arguments that cannot be
pickled deterministically (e.g. `wfl.generate_configs.minim.run` and the
`initializer=np.random.seed` argument) need to specifically exclude that
argument (obviously only if ignoring it for the purpose of detecting
duplicate submission is indeed correct).  All functions already ignore the
`outputs` `OutputSpec` argument.

```{warning}
The hashing mechanism is only designed for interrupted runs, and does
not detect changes to the called function (or to any functions that
function calls).  If the code is being modified, the user should erase the
`ExPyRe` staged job directories, and clean up the `sqlite` database file,
before rerunning.  Using a per-project `_expyre` directory makes this
easier, since the database file can simply be erased, otherwise the `xpr` command
line tool needs to be used to delete the previously created jobs.

Note that this is only relevant to incomplete autoparallelized
operations, since any completed operation (once all the remote job outputs have
been gathered into the location specified in the `OutputSpec`) no longer depends on 
anything `ExPyRe`-related.  See also the warning in the 
`OutputSpec` [documentation](overview.configset).
```

## WFL\_EXPYRE\_INFO syntax

The `WFL_EXPYRE_INFO` variable contains a JSON or the name of a file that contains a JSON.  The JSON encodes a dict with keys
indicating particular function calls, and values containing arguments for constructing 
[`RemoteInfo`](wfl.autoparallelize.remoteinfo.RemoteInfo) objects.


### Keys

Each key consist of a comma separated list of `remote_label` or `"end_of_path_to_file::function_name"`.  

If `remote_label` is used, its value needs to be given to the parallelized function's `autopara_info` as part of `AutoparaInfo` class. This allows for finer control, for example when there are multiple instances of a given parallelizable function, but they need to be handled differently. 

The list needs to match the _end_ of the stack trace, i.e. the first item matches the outermost (of the `len(str.split(','))` comma separate items specified) calling function, the second item matches the function that was called by it, etc., down to the final item matching the innermost function (not including the actual `autoparallelize` call). Each item in the list needs to match the _end_ of the file path, followed by a `:`, followed by the function name in that file.

For example, to parallelize only the call to `minin.run(...)` from `gap_rss_iter_fit.py` `do_MD_bulk_defect_step(...)`, the key could be set to ```gap_rss_iter_fit.py::do_MD_bulk_defect_step,generate_configs/minim.py::run```

### Values

Each value consists of a dict that will be passed to the `RemoteInfo` constructor as its `kwargs`.


## Dividing items into and parallelising within jobs

When using the autoparallelized loop remote job functionality, the number of items from the iterable assigned to each job is set by the `num_inputs_per_queued_job` parameter of the `RemoteInfo` object (normally set by `WFL_EXPYRE_INFO`).  If positive, it directly specifies the number, but if negative, `-1 * num_inputs_per_queued_job * num_inputs_per_python_subprocess` items will be packaged, where `num_inputs_per_python_subprocess` is the value for the underlying pool-based (`WFL_NUM_PYTHON_SUBPROCESSES`) parallelization.

Note that by default the remote job will set
- `WFL_NUM_PYTHON_SUBPROCESSES=${EXPYRE_NUM_CORES_PER_NODE}`

If this needs to be overridden, set the `env_vars` parameter of the `RemoteInfo` object

## Testing

The remote running framework can be tested with `pytest`.  To enable
these pass the `--runremote` to `pytest`.  In addition, since some
queuing systems require that jobs be submitted from a globally available
directory, the flag `--basetemp $HOME/some_path` must be passed.

The user must provide configuration information about their queueing
system.  By default, all tests that actually require a remote running system
will run on every system defined in `$HOME/.expyre/config.json`.  The
`EXPYRE_PYTEST_SYSTEMS` can define a regexp that is used to filter those.

Optional env vars:
 - `EXPYRE_PYTEST_SYSTEMS`: regexp to filter systems in `$HOME/.expyre/config.json` that will
   be used for testing.
 - `WFL_PYTEST_EXPYRE_INFO`: dict of fields to _add_ to `RemoteInfo` object when doing high
   level (`autoparallelize`, `gap_fit`) remote run tests.

### Pytest with remote run example
Running a maximally complete set of tests with somehwat verbose output (also need `pw.x`
in path):
```
> env EXPYRE_PYTEST_SYSTEMS='(local_sys|remote_sys)' \
      WFL_PYTEST_BUILDCELL=path_to_buildcell  \
      VASP_COMMAND=path_to_vasp VASP_COMMAND_GAMMA=path_to_vasp_gamma PYTEST_VASP_POTCAR_DIR=path_to_potcars \
      ASE_ORCA_COMMAND=path_to_orca \
      pytest -s --basetemp $HOME/pytest --runremote --runslow -rxXs
```
