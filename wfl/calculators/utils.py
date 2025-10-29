from pathlib import Path

from wfl.utils.file_utils import clean_dir

def clean_rundir(rundir, keep_files, default_keep_files, calculation_succeeded):
    """clean up a run directory from a file-based calculator

    Parameters
    ----------
    rundir: str
        path to run dir
    keep_files: 'default' / iterable(file_glob) / '*' / bool / None
        files to keep, None or False for nothing, '*' or True for all
    default_keep_files: list(str)
        files to keep if keep_files == 'default' or calculation_succeeded is False
        and keep_files is not True
    calculation_succeeded: bool
    """
    if isinstance(keep_files, str):
        if keep_files == 'default':
            keep_files = default_keep_files
        elif keep_files == '*':
            keep_files = [keep_files]
        else:
            raise ValueError(f"str keep_files can only be 'default' or '*', not '{keep_files}'")

    # now keep_files should either be True, evaluate to False, or an iterable

    # calculation failed, default optionally union with keep_files
    if not calculation_succeeded:
        if not keep_files:
            keep_files = default_keep_files
        elif keep_files is not True:
            keep_files = set(default_keep_files) | set(keep_files)
        # else True

    clean_dir(rundir, keep_files, force=False)
