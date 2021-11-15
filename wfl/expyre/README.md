## Overview

A class for evaluating python functions via a queuing system, aimed at
functions that take all their inputs as arguments (not files) and return
all their outputs in the return values.

Evaluation can take place on the same system as the main python process
that is calling the functions, or on a remote system (via passwordless
ssh).

## Basic use

### Example code for evaluating a function

```
import numpy as np
from expyre.func import ExPyRe

# input for function
array_to_sum = np.ones((100, 50))

# resources: run for max 1 hour on 1 node, on a bigmem partition node
res = {'max_time': '1h', 'n': (1, 'nodes'), 'partitions': '.*bigmem'}

# create a task to sum array over axis=1
xpr = ExPyRe('test_task', np.sum, args=[array_to_sum], kwargs={'axis': 1})

# submit job
xpr.start(resources=res, system_name='local')

# gather results
try:
    array_sum = xpr.get_results()
    assert np.max(np.abs(array_sum - [50.0] * 100)) == 0.0
except TimeoutError:
    print('job did not run in alloted time')

# mark as processed in jobs db in case of restarts
xpr.mark_processed()
```

Format of resources dict is
- `"max_time"`: maximum runtime (format as in configuration below)
- `"n"`: tuple(int, str) int number of str `"nodes"` or `"tasks"`
- `"partitions"`: str or list(str) with regexps that match entire partition names (see configuration below)

Additional possible keys
- `"max_mem_per_task"`: int or str max memory per task (format as in configuration below)
- `"ncores_per_task"`: int cores per task

#### restarting if main process was interrupted while waiting for jobs to finish

The simplest approach is to rely on the automated pickle/hash-based
identification of repeated jobs.  Unless the default
`try_restart_from_prev=True` is overridden when calling `ExPyRe(...)`
constructor, it will try to recreate the job if it seems identical to
a previous one.  A job is identical if the hash of the pickles are identical
for all of the following:
- function
- input arguments (except those listed in `hash_ignore`, for example any output-only arguments)
- input file names
- input file contents

The job will be recreated if the hash matches a job that
exists in the JobsDB and has status compatible with returning results
(i.e. not failed or cleaned.) "Processed" status means that the function
responsible for creating these jobs have already used the results, marked
jobs as "processed" and returned. These jobs will not be recreated and
files associated with them may be safely cleaned up. With this mechanism the
sequence of calling the restart is identical to that of the original calls.
The constructor will recreate the `ExPyRe` object, the `start` call will
submit it if necessary, and the `get_results` will sync remote files if needed and
return the unpickled results.

Another approach, applicable to jobs that have been started, is to
manually recreate them from the JobsDB database.  Syntax is
```
db = expyre.config.db
xprs = ExPyRe.from_jobsdb(db.jobs(name='task_1'))
for xpr in xprs:
    res = xpr.get_results()
    # do something with res
```
The `db.jobs(...)` call can filter with regexps on `status`, `name`,
`id`, or `system`.

### Configuration

The description of the available systems and resources is in a `config.json` file.

#### config.json file format

The top level structure is a dict with one `"systems"` key with a dict value, containing
one key-value pair for each system. The key is the name of the system (used to select
the system when evaluating the function).  The value is a dict with the following keys:
- `"host"`: string hostname, with optional `username@`, or `null` for localhost without ssh.
- `"scheduler"`: string indicating type of scheduler, currently `"slurm"`, `"pbs"` or `"sge"`
- `"commands"`: optional list(str) with commands to run at beginning of every script on system,
   usually for things that set up the runtime such as environment modules (e.g. `module load vasp`)
- `"header"`: list(str) queuing system header lines.  Actual header is created by applying string
              formatting, i.e.  `str.format(**kwargs)`, replacing substrings such as `{nnodes}`.
              Available keys in `kwargs` which are normally used in this template:
  - `"nnodes"`: int total number of nodes
  - `"ncores_per_node"`: int number of cores per node
  - `"tot_ncores"`: int total number of cores

  Note that if `partial_node=True` is passed to `find_nodes` and the total number of cores is less
  than the number per node, `tot_ncores` and `ncores_per_node` are *not* rounded up to an entire node.

  Additional keys that are not normally used for the header (having to do with two level parallelism where
  each task may be allocated more than 1 node):
  - `"tot_ntasks"`: int total number of tasks (`tot_ncores / ncores_per_task` or `nnodes * ntasks_per_node`)
  - `"ncores_per_task"`: int cores per task (specified in resources)
  - `"ntasks_per_node"`: int tasks per node (`ncores_per_node / ncores_per_task`)

  Additional keys that are used by the internally generated parts of the header:
  - `"id"`: str (supposed to be guaranteed to be unique among current jobs within project) job id
  - `"partition"`: str partition/queue/node type
  - `"max_time_HMS"`: str max runtime in `hours:minutes:seconds` format
- `"partitions"`: dict with partitions/queues/node-type names as keys and dict of properties as values.
Property dict includes
  - `"ncores"`: int number of cores per node
  - `"max_time"`: max time, int for seconds, str for `"<N>[smhd]"` (case insensitive) or `"<dd>-<hh>:<mm>:<ss>"`
  (with leading parts optional, so N1:N2 is N1 minutes + N2 seconds)
  - `"max_mem"`: max mem per node, int for kB, str for `"<N>[kmgt]b?"` (case insensitive).
- `"remsh_cmd"`: optional string remote shell command, default `"ssh"`
- `"no_default_header"`: bool, default false disable automatic setting of queuing system header for job name,
    partition/queue, max runtime, and stdout and stderr files
- `"rundir"`: string for a path where the remote jobs should be run

#### Configuration file location and project scope

The `expyre` root dir is the location of the jobs database and staging directories for each function call.
By setting this appropriately the namespace associated with different projects can be separated.

If the `EXPYRE_ROOT` env var is set, it is used as location for `config.json` as well as the `expyre` root.
If it is not set, then `$HOME/.expyre` is first read, then starting from `$CWD/_expyre` and going up
one directory at a time.  The expyre root is the directory closest to `$CWD` which contains a `_expyre`
subdirectory.  The configuration consists of the content of `$HOME/.expyre/config.json`, modified by any
entries that are present in `<dir>/_expyre/config.json`
(with `home_systems[sys_name].dict.update(local_systems[sys_name])` on the dict associated with each
system).  New systems are added, and a system dict value of `null` removes the system.  Currently within
each system each key-value pair is overwritten, so you cannot directly disable one partition, only redefine
the entire `"partitions"` dict.

#### config.json example

```
{ "systems": {
    "local": { "host": "localhost",
        "remsh_cmd": "/usr/bin/ssh",
        "scheduler": "slurm",
        "commands": [ "module purge", "module load python/3 compilers/gnu lapack ase quip vasp" ],
        "header": ["#SBATCH --nodes={nnodes}",
                   "#SBATCH --ntasks={tot_ncores}",
                   "#SBATCH --ntasks-per-node={ncores_per_node}"],
        "partitions": { "node16_old,node16_new": { "ncores" : 16, "max_time" : null, "max_mem" : "60GB" },
                        "node36":                { "ncores" : 36, "max_time" : null, "max_mem" : "180GB" },
                        "node32":                { "ncores" : 32, "max_time" : null, "max_mem" : "180GB" },
                        "node56_bigmem":         { "ncores" : 56, "max_time" : "48:00:00", "max_mem" : "1500GB" },
                        "node72_bigmem":         { "ncores" : 72, "max_time" : "48h", "max_mem" : "1500GB" }
        }
    }
}
```

For this system:
- Connect with `/usr/bin/ssh` to localhost
- use slurm commands to submit jobs
- do some env mod stuff in each job before running task
- use built-in header for job name, partition, time, stdout/stderr, and + specified 3 lines to select number of nodes
- define 5 partitions (names in slurm `--partition` format), with varying numbers of cores, memory, and time limit
    on the two `_bigmem` ones (same time, specified in different formats as an example).

## environment variables

#### at expyre runtime
- `EXPYRE_RETRY`: 'n t' number of tries `n` and wait time in seconds `t` if subprocess run fails, default '3 5'
- `EXPYRE_ROOT`: override path to root directory for `config.json`, `JobsDB`, and stage directories
- `EXPYRE_RSH`: default remote shell command if not specified in system configuration, overall default `ssh`
- `EXPYRE_SYS`: default system to start remote functions on, if not specified in call to `ExPyRe.start()`
- `EXPYRE_TIMING_VERBOSE`: print trace (to stderr) with timing info to determine what operation is taking time

#### available in submitted job scripts
- `EXPYRE_NNODES`
- `EXPYRE_TOT_NCORES`
- `EXPYRE_TOT_NTASKS`
- `EXPYRE_NCORES_PER_NODE`
- `EXPYRE_NTASKS_PER_NODE`
- `EXPYRE_NCORES_PER_TASK`

#### only for pytest
- `EXPYRE_PYTEST_CLEAN`: if set, do a real cleaning of directories in pytest `test_func.py`, otherwise only a dry run
- `EXPYRE_PYTEST_SSH`: path to ssh to use (instead of `/usr/bin/ssh`) in pytest `test_subprocess.py` (all higher level
    tests use `remsh_cmd` item for system in `config.json`
- `EXPYRE_PYTEST_SYSTEMS`: regexp to use to filter systems from those available in `$HOME/.expyre/config.json`
- `EXPYRE_PYTEST_QUEUED_JOB_RESOURCES`: JSON or filename with JSON that defines an array of two dicts, each with Resources
    kwargs so that first one is a small job, and second one is large enough that it's guaranteed to be queued once
    the first has been submitted (e.g. using _all_ available nodes).

## CLI

There is a command line interface, currently only for simple interaction with the jobs database. Available commands
- `xpr ls`: list jobs
- `xpr rm`: delete jobs (and optionally local/remote stage directories)
- `xpr db_unlock` (unlock jobs database if it was locked when a process crashed)

Use `xpr --help` for more info.


## Code structure

[more details coming soon]
