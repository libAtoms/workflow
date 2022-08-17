# Workflow infrastructure

## ConfigSet and OutputSpec, thin I/O layers for input and output

`ConfigSet` and `OutputSpec` are python classes defined in `configset.py`.
```python
from wfl.configset import ConfigSet, OutputSpec
```
`ConfigSet` can encapsulate one or multiple lists of `ase.atoms.Atoms` objects,
or reference to stored sets of configuration in files or ABCD databases.
It can function as an iterator over all configs in the input, or iterate over groups of them
according to the input definition with the `ConfigSet().group_iter()` method.
The `ConfigSet` must be initialized with its inputs and indices for them.

`OutputSpec` works as the output layer, can be used for writing results into it during
iterations, but the actual writing is only happening when the operation is closed with
`OutputSpec.end_write()`. Input mapping can be added to output into multiple files,
based on the input. This is not fully functional for putting configs into the different
outputs in a random order and repeatedly touching one.

For example, to read from two files and write corresponding configs to
two other files, use
```python
s_in = ConfigSet(input_files=['in1.xyz','dir/in2.xyz'])
s_out = OutputSpec(output_files={"in1.xyz": "out1.xyz", "in2.xyz": "out2.xyz"})
for at in s_in:
    do_some_operation(at)
    s_out.write(at, from_input_file=s_in.get_current_input_file())
s_out.end_write()
```
In this case the inputs is a list of files, and the outputs is either a single file (many -> 1)
or a mapping between equal number of input and output categories (multiple 1 -> 1).
This will not overwrite unless you also pass `force=True`.

To read from and write to ABCD database records, you can do
```python
output_tags = {'output_tag' : 'some unique value'}
s = ConfigSet(input_abcd='mongodb://localhost:27017' inputs={'input_tag' : 'necessary_input-val'},
              output_abcd='mongodb://localhost:27017', output_tags=output_tags)
```
In this case the inputs are a dict (single query, AND for each
key-value pair) or list of dict for queries (multiple queries, OR
of all the dicts), and the output tags are a dict of tags and values
to set on writing.  Unless `output_force=True`, this will refuse
to write if any config already has the output tags set (to ensure
that all the configurations written by the loop can be retrieved exactly, for
passing to the next stage in the pipeline).  The outputs can be retrieved by
```python
abcd.get_atoms(output_tags)
```
## running wfl functions as independently queued jobs


Operations that have been wrapped in `autoparallelize` can be split into independent jobs with minimal
coding.  A `config.json` file must describe the available queuing sytem resources (see
[expyre README](https://github.com/libAtoms/ExPyRe#readme)), and a JSON file (content or path in
env var `WFL_EXPYRE_INFO`) describes the resources needed by any `autoparallelize` call that
should be executed this way.  Any remote machine to be used requires that the `wfl` python
module be installed.  If needed, commands needed to make this module available (e.g. setting `PYTHONPATH`)
can be set on a per-machine basis in the `config.json` file mentioned below.

In addition, `wfl.fit.gap_simple`, `wfl.fit.gap_multistage`, and `wfl.fit.ace` have been wrapped, as a single
job each.  The GAP simple fit is controlled by the `WFL_GAP_SIMPLE_FIT_REMOTEINFO` env var.  Setting
this variable will also lead to the multistage fit submitting each simple fit as its own job.
In addition, the multistage fit can be turned into a single job containing all the stages
with the `WFL_GAP_MULTISTAGE_FIT_REMOTEINFO` env var.  In principle, doing each simple fit
as its own job could enable running committee fits in parallel, but that is not supported right now.
The env var `WFL_ACE_FIT_REMOTEINFO` is used for ACE fits.

[NOTE: now that the multistage fit does very little other than the repeated simple fitting, does
it need its own level of remote job execution]

### Example

The workflow (`do_workflow.py`) is essentially identical to what you'd otherwise construct:
```
from wfl.configset import ConfigSet, OutputSpec
from wfl.generate_configs.minim import run

from ase.calculators.vasp import Vasp

infile = "structures.xyz"
ci = ConfigSet(input_files=infile)
co = OutputSpec(output_files='relaxed.' + infile, force=True, all_or_none=True)

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

In addition a `config.json` file (see example in [expyre README]https://github.com/libAtoms/ExPyRe#readme))
with available systems needs to be created
(usually in `$HOME/.expyre/`, otherwise in the `_expyre` directory
mentioned below), and a directory called `some/path/_expyre/` (note
the initial `_`, not `.`, so it is more visible) can optionally be created at
the directory hierarchy level that indicates the scope of the project,
to separate the jobs database from any other project.

Restarts are supposed to be handled automatically - if the workflow script is
interrupted, just rerun it.  If the entire `autoparallelize` call is complete,
the default of `force=True, all_or_none=True` for `OutputSpec()` will allow
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

### WFL\_AUTOPARA\_REMOTEINFO syntax

The `WFL_EXPYRE_INFO` variable contains a JSON or the name of a file that contains a JSON.  The JSON encodes a dict with keys
indicating particular function calls, and values containing arguments for constructing [`RemoteInfo`](wfl/pipeline/utils.py) objects.

#### keys

Each key consist of a comma separated list of `"end_of_path_to_file::function_name"`.  The list needs to match the _end_ of the stack
trace, i.e. the first item matches the outermost (of the `len(str.split(','))` comma separate items specified) calling function, the second item matches
the function that was called by it, etc., down to the final item matching the innermost function (not including the actual `autoparallelize` call).
Each item in the list needs to match the _end_ of the file path, followed by a `:`, followed by the function name in that file.

For example, to parallelize only the call to `minin.run(...)` from `gap_rss_iter_fit.py` `do_MD_bulk_defect_step(...)`, the key could be set to
```gap_rss_iter_fit.py::do_MD_bulk_defect_step,generate_configs/minim.py::run```

#### values

Each value consists of a dict that will be passed to the `RemoteInfo` constructor as its `kwargs`.

### Dividing items into and parallelising within jobs

When using the autoparallelized loop remote job functionality, the number
of items from the iterable assigned to each job is set by the
`num_inputs_per_queued_job` parameter of the `RemoteInfo` object (normally set by
`WFL_EXPYRE_INFO`).  If positive, it directly specifies the
number, but if negative, `-1 * num_inputs_per_queued_job * num_inputs_per_python_subprocess` items will be
packaged, where `num_inputs_per_python_subprocess` is the value for the underlying pool-based
(`WFL_NUM_PYTHON_SUBPROCESSES`) parallelization.

Note that by default the remote job will set
- `WFL_NUM_PYTHON_SUBPROCESSES=${EXPYRE_NCORES_PER_NODE}`

If this needs to be overridden, set the `env_vars` parameter of the `RemoteInfo` object

### testing

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

#### pytest with remote run example
Running a maximally complete set of tests with somehwat verbose output (also need `pw.x`
in path):
```
> env EXPYRE_PYTEST_SYSTEMS='(local_sys|remote_sys)' \
      WFL_PYTEST_BUILDCELL=path_to_buildcell  \
      VASP_COMMAND=path_to_vasp VASP_COMMAND_GAMMA=path_to_vasp_gamma PYTEST_VASP_POTCAR_DIR=path_to_potcars \
      ASE_ORCA_COMMAND=path_to_orca \
      pytest -s --basetemp $HOME/pytest --runremote --runslow -rxXs
```
