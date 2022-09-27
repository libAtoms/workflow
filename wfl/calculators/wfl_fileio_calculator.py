from pathlib import Path
import shutil
import tempfile

from .utils import clean_rundir

class WFLFileIOCalculator():
    """Mixin class implementing some methods that should be available to every
    WFL calculator that does I/O via files, i.e. DFT calculators

    Parameters
    ----------
    keep_files: bool / None / "default" / list(str), default "default"
        what kind of files to keep from the run
            True, "*" : everything kept
            None, False : nothing kept
            "default"   : default list, varies by calculator, usually only ones needed for NOMAD uploads
            list(str)   : list of file globs to save
    rundir: str / Path, default 'run\_<calculator>\_'
        Run directory name prefix (or full name - see reuse_rundir)
    reuse_rundir: bool, default False
        Treat rundir as a fixed directory, rather than a prefix to a unique dir name.
        WARNING: Do not set rundir to an existing directory with other files, because they
        may be deleted by clean_rundir() at the end of the calculation.
    workdir: str / Path, default . at calculate time
        Path in which rundir will be created.
    scratchdir: str / Path, default None
        temporary directory to execute calculations in and delete or copy back results (set by
        "keep_files") if needed.  For example, directory on a local disk with fast file I/O.
    kwargs: dict
        remaining superclass constructor kwargs
    """

    def __init__(self, /, keep_files, rundir, reuse_rundir, workdir, scratchdir, **kwargs):
        if "directory" in kwargs:
            raise ValueError("Cannot pass directory argument")

        super().__init__(**kwargs)

        self._wfl_keep_files = keep_files

        self._wfl_rundir = Path(rundir)
        self._wfl_reuse_rundir = reuse_rundir
        self._wfl_workdir = Path(workdir)
        self._wfl_scratchdir = Path(scratchdir) if scratchdir is not None else None


    def setup_rundir(self):
        # set rundir to where we want final results to live
        rundir_path = self._wfl_workdir / self._wfl_rundir.parent
        rundir_path.mkdir(parents=True, exist_ok=True)
        if self._wfl_reuse_rundir:
            self._cur_rundir = rundir_path / self._wfl_rundir.name
            self._cur_rundir.mkdir(exist_ok=True)
        else:
            self._cur_rundir = Path(tempfile.mkdtemp(dir=rundir_path, prefix=self._wfl_rundir.name))

        # set self.directory to where we want the calculation to actully run
        if self._wfl_scratchdir is not None:
            directory = self._wfl_scratchdir / (str(self._cur_rundir.resolve()).replace("/", "", 1))
            directory.mkdir(parents=True, exist_ok=True)
            self.directory = directory
        else:
            self.directory = self._cur_rundir


    def clean_run(self, _default_keep_files, calculation_succeeded):
        clean_rundir(self.directory, self._wfl_keep_files, _default_keep_files, calculation_succeeded)
        if self._wfl_scratchdir is not None:
            for f in Path(self.directory).glob("*"):
                shutil.move(f, self._cur_rundir)
