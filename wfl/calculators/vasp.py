"""
VASP calculator

"""

import os
import shutil

import json

import numpy as np

from ase.calculators.calculator import all_changes
from ase.calculators.vasp.vasp import Vasp as ASE_Vasp

from .wfl_fileio_calculator import WFLFileIOCalculator
from wfl.utils.save_calc_results import save_calc_results
from .kpts import universal_kspacing_n_k

from ase.calculators.vasp.create_input import float_keys, exp_keys, string_keys, int_keys, bool_keys
from ase.calculators.vasp.create_input import list_int_keys, list_bool_keys, list_float_keys, special_keys, dict_keys


# NOMAD compatible, see https://nomad-lab.eu/prod/rae/gui/uploads
_default_keep_files = ["POSCAR", "INCAR", "KPOINTS", "OUTCAR", "vasprun.xml", "vasp.out"]
_default_properties = ["energy", "forces", "stress"]

class Vasp(WFLFileIOCalculator, ASE_Vasp):
    """Extension of ASE's Vasp calculator that can be used by wfl.calculators.generic

    Notes
    -----
    "directory" argument cannot be present. Use rundir_prefix and workdir instead.

    "command_gamma" (or ASE_VASP_COMMAND_GAMMA) is used when non-periodic cells or large
    enough kspacing are detected

    "pp" defaults to ".", so VASP_PP_PATH env var is absolute path to "<elem name>/POTCAR" files

    Parameters
    ----------
    keep_files: bool / None / "default" / list(str), default "default"
        what kind of files to keep from the run
            True, "*" : everything kept
            None, False : nothing kept
            "default"   : only ones needed for NOMAD uploads ('POSCAR', 'INCAR', 'KPOINTS', 'OUTCAR', 'vasprun.xml', 'vasp.out')
            list(str)   : list of file globs to save
    rundir_prefix: str / Path, default 'run\_VASP\_'
        Run directory name prefix
    workdir: str / Path, default . at calculate time
        Path in which rundir will be created.
    scratchdir: str / Path, default None
        temporary directory to execute calculations in and delete or copy back results (set by
        "keep_files") if needed.  For example, directory on a local disk with fast file I/O.
    command_gamma: str, default None
        command to use when gamma-only calculations are detected (e.g. large kspacing, or pbc=False)
    **kwargs: arguments for ase.calculators.vasp.vasp.Vasp
        remaining arguments to ASE's Vasp calculator constructor
    """

    # default value of wfl_num_inputs_per_python_subprocess for calculators.generic,
    # to override that function's built-in default of 10
    wfl_generic_num_inputs_per_python_subprocess = 1

    # note that we also have a default for "pp", but that has to be handlded separately
    default_parameters = ASE_Vasp.default_parameters.copy()
    default_parameters.update({
        "ismear": 0,
        "lwave": False,
        "lcharg": False,
    })

    def __init__(self, keep_files="default", rundir_prefix="run_VASP_",
                 workdir=None, scratchdir=None, command_gamma=None,
                 **kwargs):

        # get initialparams from env var
        kwargs_use = {}
        if 'WFL_VASP_KWARGS' in os.environ:
            try:
                kwargs_use = json.loads(os.environ['WFL_VASP_KWARGS'])
            except json.decoder.JSONDecodeError:
                with open(os.environ['WFL_VASP_KWARGS']) as fin:
                    kwargs_use = json.load(fin)

        # override with explicitly passed in values
        kwargs_use.update(kwargs)

        # pp is not handled by Vasp.default_parameters, because that only includes
        # parameters that are in INCAR
        if "pp" not in kwargs_use:
            kwargs_use["pp"] = "."

        self._command_gamma = command_gamma

        self.universal_kspacing = kwargs_use.pop("universal_kspacing", None)

        self.multi_calculation_kwargs = kwargs_use.pop("multi_calculation_kwargs", None)
        if self.multi_calculation_kwargs is None:
            self.multi_calculation_kwargs = [{}]

        self.debug = kwargs_use.pop("debug", False)

        # WFLFileIOCalculator is a mixin, will call remaining superclass constructors for us
        super().__init__(keep_files=keep_files, rundir_prefix=rundir_prefix,
                         workdir=workdir, scratchdir=scratchdir, **kwargs_use)

    def per_config_setup(self, atoms):
        # ASE Vasp calculator complains if pbc=F
        self._orig_pbc = atoms.pbc.copy()
        atoms.pbc[:] = True

        # VASP requires periodic cells with non-zero cell vectors
        if np.dot(np.cross(atoms.cell[0], atoms.cell[1]), atoms.cell[2]) < 0.0:
            t = atoms.cell[0].copy()
            atoms.cell[0] = atoms.cell[1]
            atoms.cell[1] = t
            self._permuted_a0_a1 = True
        else:
            self._permuted_a0_a1 = False

        # switch to gamma pt only (including executable if available) if fully nonperiodic
        self._orig_command = self.command
        self._orig_kspacing = self.float_params["kspacing"]
        self._orig_kgamma = self.bool_params["kgamma"]
        use_gamma_exec = np.all(~self._orig_pbc)
        # gamma centered if kgamma is undefined (default) or True
        if self._orig_kgamma is None or self._orig_kgamma:
            try:
                n_k = np.maximum(1, np.ceil(np.linalg.norm(atoms.cell.reciprocal(), axis=1) * 2.0 * np.pi / self._orig_kspacing))
                use_gamma_exec |= np.all(n_k == 1)
            except TypeError:
                pass

        if use_gamma_exec:
            # set command
            if self._command_gamma is not None:
                # from constructor argument that was saved
                command_gamma = self._command_gamma
            else:
                # from env var
                command_gamma = None
                for env_var in self.env_commands:
                    if env_var + "_GAMMA" in os.environ:
                        command_gamma = os.environ[env_var + "_GAMMA"]
                        break
            if command_gamma is not None:
                self.command = command_gamma

            # explicitly set k-points for nonperiodic systems
            self.float_params["kspacing"] = 1.0e8
            self.bool_params["kgamma"] = True


    def per_config_restore(self, atoms):
        # undo pbc change
        atoms.pbc[:] = self._orig_pbc

        # undo cell vector permutation
        if self._permuted_a0_a1:
            t = atoms.cell[0].copy()
            atoms.cell[0] = atoms.cell[1]
            atoms.cell[1] = t

        # undo nonperiodic related changes
        self.command = self._orig_command
        self.float_params["kspacing"] = self._orig_kspacing
        self.bool_params["kgamma"] = self._orig_kgamma


    def param_dict_of_key(self, k):
        if k in float_keys:
            d = self.float_params
        elif k in exp_keys:
            d = self.exp_params
        elif k in string_keys:
            d = self.string_params
        elif k in int_keys:
            d = self.int_params
        elif k in bool_keys:
            d = self.bool_params
        elif k in list_int_keys:
            d = self.list_int_params
        elif k in list_bool_keys:
            d = self.list_bool_params
        elif k in list_float_keys:
            d = self.list_float_params
        elif k in special_keys:
            d = self.special_params
        elif k in dict_keys:
            d = self.dict_params
        return d


    def calculate(self, atoms=None, properties=_default_properties, system_changes=all_changes):
        """Do the calculation. Handles the working directories in addition to regular
        ASE calculation operations (writing input, executing, reading_results)"""

        if atoms is not None:
            self.atoms = atoms.copy()

        # hack to disable parallel I/O, since that makes VASP's I/O break if
        # this script is being run with mpirun
        from ase.parallel import world, DummyMPI
        orig_world_comm = world.comm
        world.comm = DummyMPI()

        # make sure VASP_PP_PATH is set to a sensible default
        orig_VASP_PP_PATH = os.environ.get("VASP_PP_PATH")
        if orig_VASP_PP_PATH is None:
            if self.input_params['pp'].startswith("/"):
                os.environ["VASP_PP_PATH"] = "/."
            else:
                os.environ["VASP_PP_PATH"] = "."

        # do this before mangling atoms.pbc values
        if self.universal_kspacing is not None:
            assert all([k in ["kspacing", "kgamma"] for k in self.universal_kspacing]), "Unknown field in 'universal_kspacing'"
            self.kpts = universal_kspacing_n_k(cell=atoms.cell, pbc=atoms.pbc, kspacing=self.universal_kspacing["kspacing"])
            self.input_params["gamma"] = self.universal_kspacing.get("kgamma", True)

        self.per_config_setup(atoms)

        # from WFLFileIOCalculator
        self.setup_rundir()

        atoms.info["vasp_rundir"] = str(self._cur_rundir)

        if self.float_params["encut"] is None:
            raise RuntimeError("Refusing to run without explicit ENCUT")
        # should we require anything except ENCUT?

        # clean up old failure notes
        atoms.info.pop('DFT_FAILED_VASP', None)
        calculation_succeeded = False
        try:
            for multi_i, multi_calculation_kwargs_set in enumerate(self.multi_calculation_kwargs):
                if self.debug: print(f"multi_calculation instance {multi_i}")
                if multi_calculation_kwargs_set is None:
                    multi_calculation_kwargs_set = {}
                prev_vals = {}
                prev_dicts = {}
                for k, v in multi_calculation_kwargs_set.items():
                    if self.debug: print("  multi_calculation override", k, v)
                    # figure out which dict this param goes in
                    d = self.param_dict_of_key(k)
                    # save prev value and dict it's in (for easy restoring later)
                    prev_vals[k] = d[k]
                    prev_dicts[k] = d
                    if self.debug: print("  multi_calculation save prev val", prev_vals[k])
                    # set new value
                    d[k] = v
                super().calculate(atoms=atoms, properties=properties, system_changes=system_changes)
                if self.debug: shutil.copy(self._cur_rundir / "OUTCAR", self._cur_rundir / f"OUTCAR.{multi_i}")
                # restore previous values
                for k, v in multi_calculation_kwargs_set.items():
                    if self.debug: print("  multi_calculation restoring from prev val", k, prev_vals[k])
                    prev_dicts[k][k] = prev_vals[k]
            calculation_succeeded = True
            # save results here (if possible) so that save_calc_results() called by calculators.generic
            # won't trigger additional calculations due to the ASE caching noticing the change in pbc
            if "__calculator_output_prefix" in atoms.info:
                save_calc_results(atoms, prefix=atoms.info["__calculator_output_prefix"], properties=properties)
                atoms.info["__calculator_results_saved"] = True
        except Exception as exc:
            atoms.info['DFT_FAILED_VASP'] = True
            raise exc
        finally:
            # from WFLFileIOCalculator
            self.clean_rundir(_default_keep_files, calculation_succeeded)

            self.per_config_restore(atoms)

            # undo env VASP_PP_PATH
            if orig_VASP_PP_PATH is not None:
                os.environ["VASP_PP_PATH"] = orig_VASP_PP_PATH

            # undo communicator mangling
            world.comm = orig_world_comm
