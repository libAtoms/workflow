import os
import shutil
from glob import glob


def clean_dir(directory, keep_files, force=False):
    """Clean a run directory and keep only the specified files

    Parameters
    ----------
    directory : directory to be cleaned
    keep_files: bool or iterable(file_globs)
        What to keep in rundir when done:
            - True : keep everything
            - iterable : ONLY files matching these globs if they exist
            - evaluates to False : nothing
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

    if isinstance(keep_files, str):
        # common error that will not be caught below
        raise ValueError(f"keep_files must not be str '{keep_files}'")

    # operations
    if keep_files is True:
        # bool True, keep all
        return
    elif not keep_files:
        # evaluates to False, keep none
        shutil.rmtree(directory)
    else:
        # keep_files should be an iterable
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
