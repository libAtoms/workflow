import os
import shutil
from glob import glob


def clean_dir(directory, keep_files, force=False):
    """Clean a run directory and keep only the specified files

    Parameters
    ----------
    directory : directory to be cleaned
    keep_files: bool or list(filenames) or str
        What to keep in rundir when done:
            - list(filenames) : ONLY these filenames if they exist
            - True or "*" : everything - does nothing
            - False or None : remove everything, or anything evaluating to False in if
    force : bool, default = False
        fail if directory does not exist

    Returns
    -------

    """

    # if the dir is non-existent
    if not os.path.isdir(directory):
        if force:
            raise FileNotFoundError(f"No directory to be cleaned {directory}")
        else:
            return

    # defaults
    if keep_files is None:
        keep_files = False
    elif keep_files == "*":
        keep_files = True
    elif isinstance(keep_files, str):
        keep_files = [keep_files]

    # operations
    if isinstance(keep_files, bool) and keep_files:
        return
    elif not keep_files:
        # lets None and anything else evaluating to False
        shutil.rmtree(directory)
    elif isinstance(keep_files, (list, tuple, set)):
        extended_keep_files = []
        for glob_index in keep_files:
            extended_keep_files.extend(glob(os.path.join(directory, glob_index)))
        extended_keep_files = set(extended_keep_files)

        for run_file in os.listdir(directory):
            abs_fn = os.path.join(directory, run_file)
            if abs_fn not in extended_keep_files:
                if os.path.isfile(abs_fn):
                    os.remove(abs_fn)
                else:
                    shutil.rmtree(abs_fn)
    else:
        raise RuntimeError('Got unknown type or value for keep_rundir \'{}\''.format(keep_files))
