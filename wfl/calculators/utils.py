from wfl.utils.file_utils import clean_dir

def clean_rundir(rundir, keep_files, default_keep_files, calculation_succeeded):
    """clean up a run directory from a file-based calculator

    Parameters
    ----------
    rundir: str
        path to run dir
    keep_files: 'default' / list(str) / '*' / bool / None
        files to keep, None or False for nothing, '*' or True for all
    default_keep_files: list(str)
        files to keep if keep_files == 'default' or calculation_succeeded is False
    calculation_succeeded: bool
    """
    if not calculation_succeeded:
        clean_dir(rundir, set(default_keep_files) | set(keep_files if keep_files else []), force=False)
    elif keep_files == 'default':
        clean_dir(rundir, default_keep_files, force=False)
    elif not keep_files:
        clean_dir(rundir, False, force=False)
    else:
        clean_dir(rundir, keep_files, force=False)

def parse_genericfileio_profile_argv(argv):
    """Parse a command provided as a conventional argv into the separate
    structures that generic file-io calculators use to construct their Profile

    Parameters
    ----------
    argv: list(str)
        command to execute, split into separate arguments (e.g. using shlex.split?)

    Returns
    -------
    binary: str binary to execute
    parallel_info: dict with parallel info, in particular "binary" for mpirun/mpiexec/srun etc,
                   and additional fields to reconstruct rest of command line (all fake, depending
                   on details of how ASE constructs the final command line
    """
    binary = argv[-1]
    parallel_info = None
    if len(argv) > 1:
        # assume earlier arguments are parallel execution dependent, in particular
        # mpirun/mpiexec/srun [other mpi argument] pw_executable
        parallel_info = {"binary": argv[0]}
        for arg in argv[1:-1]:
            # add additional arguments, faked into a dict that ASE will convert into
            # a proper command line
            parallel_info[arg] = True

    return binary, parallel_info
