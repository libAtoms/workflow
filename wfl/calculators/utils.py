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


