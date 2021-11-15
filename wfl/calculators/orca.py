import os
import pathlib
import re
import shutil
import sys
import tempfile
import traceback

import ase.io
import numpy as np
from ase import units
from ase.calculators.calculator import CalculationFailed, Calculator, \
    FileIOCalculator, all_changes
from ase.calculators.orca import ORCA

from .utils import clean_rundir, save_results
from ..pipeline import iterable_loop
from ..utils.misc import atoms_to_list, chunks

__default_keep_files = ["*.inp", "*.out", "*.ase", "*.engrad", "*.xyz",
                        "*_trj.xyz"]


def evaluate(inputs, outputs,
             base_rundir=None, dir_prefix="ORCA_", keep_files="default",
             orca_kwargs=None,
             output_prefix=None, basin_hopping=False):
    """Evaluate with BasinHoppingORCA calculator

    Parameters
    ----------
    inputs: list(Atoms) / Configset_in
        input atomic configs, needs to be iterable
    outputs: list(Atoms) / Configset_out
        output atomic configs
    base_rundir: path-like, default os.getcwd()
        directory to put calculation directories into
    dir_prefix: str, default 'ORCA_'
        directory name prefix for calculations
    keep_files: "default" / bool
        what kind of files to keep from the run
            - "default : .out, .inp, .ase, .engrad is kept -- ase can read
            the results again
            - True : all files kept
            - False : none kept
    orca_kwargs: dict
        kwargs for BasinHoppingORCA calculator
    output_prefix : str, default None
        prefix for keys in the
    basin_hopping : bool, default=False
        to use basin hopping in wavefunction space, advised together with
        smearing for radicals

    Returns
    -------
    results : Configset_in
        outputs.to_ConfigSet_in()
    """
    return iterable_loop(iterable=inputs, configset_out=outputs,
                         op=evaluate_op,
                         base_rundir=base_rundir, dir_prefix=dir_prefix,
                         keep_files=keep_files, orca_kwargs=orca_kwargs,
                         output_prefix=output_prefix,
                         basin_hopping=basin_hopping)


def evaluate_basin_hopping(inputs, outputs,
                           base_rundir=None, dir_prefix="ORCA_",
                           keep_files="default", orca_kwargs=None,
                           output_prefix=None):
    """Evaluate with BasinHoppingORCA calculator

    Parameters
    ----------
    inputs: list(Atoms) / Configset_in
        input atomic configs, needs to be iterable
    outputs: list(Atoms) / Configset_out
        output atomic configs
    base_rundir: path-like, default os.getcwd()
        directory to put calculation directories into
    dir_prefix: str, default 'ORCA_'
        directory name prefix for calculations
    keep_files: "default" / bool
        what kind of files to keep from the run
            - "default : .out, .inp, .ase, .engrad is kept -- ase can read
            the results again
            - True : all files kept
            - False: none kept
    orca_kwargs: dict
        kwargs for BasinHoppingORCA calculator
    output_prefix : str, default None
        prefix for keys in the

    Returns
    -------
    results : Configset_in
        outputs.to_ConfigSet_in()
    """
    return iterable_loop(iterable=inputs, configset_out=outputs,
                         op=evaluate_op,
                         base_rundir=base_rundir, dir_prefix=dir_prefix,
                         keep_files=keep_files, orca_kwargs=orca_kwargs,
                         output_prefix=output_prefix, basin_hopping=True)


def evaluate_op(atoms, base_rundir=None, dir_prefix="ORCA_",
                keep_files="default", orca_kwargs=None,
                output_prefix="ORCA_", basin_hopping=False):
    """Evaluate with ORCA, optionally with BasinHopping

    Parameters
    ----------
    atoms: Atoms / list(Atoms)
        input atomic configs
    base_rundir: path-like, default os.getcwd()
        directory to put calculation directories into
    dir_prefix: str, default 'ORCA_'
        directory name prefix for calculations
    keep_files: "default" / bool
        what kind of files to keep from the run
            - "default : .out, .inp, .ase, .engrad is kept -- ase can read
            the results again
            - True : all files kept
            - False: none kept
    orca_kwargs: dict
        kwargs for BasinHoppingORCA calculator
    output_prefix : str, default "ORCA_"
        prefix for keys in the
    basin_hopping : bool, default=False
        to use basin hopping in wavefunction space, advised together with
        smearing for radicals

    Returns
    -------
    results: Atoms / list(Atoms)

    """
    at_list = atoms_to_list(atoms)

    if orca_kwargs is None:
        orca_kwargs = dict()

    if keep_files != "default" or not keep_files:
        keep_files = keep_files or orca_kwargs.get("keep_files", False)

    if base_rundir is None:
        # using the current directory
        base_rundir = os.getcwd()
    else:
        pathlib.Path(base_rundir).mkdir(parents=True, exist_ok=True)

    for at in at_list:
        if not basin_hopping and "scratch_path" in orca_kwargs.keys():
            # ExtendedORCA has no concept of scratch_path, but setting it
            # here we can easily go around that
            rundir = tempfile.mkdtemp(dir=orca_kwargs["scratch_path"],
                                      prefix=dir_prefix)
        elif keep_files:
            rundir = tempfile.mkdtemp(dir=base_rundir, prefix=dir_prefix)
        else:
            rundir = "."

        # specify the calculator
        if basin_hopping:
            at.calc = BasinHoppingORCA(
                **dict(orca_kwargs, directory=rundir, keep_files=keep_files))
        else:
            at.calc = ExtendedORCA(**dict(orca_kwargs, directory=rundir))

        # skip if calculation fails
        calculation_succeeded = False
        try:
            at.calc.calculate(at)
            calculation_succeeded = True
        except CalculationFailed:
            pass

        if calculation_succeeded and not basin_hopping:
            # task='opt' in ExtendedORCA performs geometry optimisation,
            # results are for the relaxed positions
            if "relaxed_positions" in at.calc.extra_results.keys():
                at.set_positions(at.calc.extra_results["relaxed_positions"])

            # reading of the normal mode data
            if "freq" in orca_kwargs.get("task", ""):
                at.info[
                    f"{output_prefix}normal_mode_eigenvalues"] = \
                    at.calc.extra_results.get(
                        "normal_mode_eigenvalues")
                for i, evec in enumerate(at.calc.extra_results.get(
                        "normal_mode_displacements")):
                    # the ORCA are taken "as is" from the file
                    at.new_array(
                        f"{output_prefix}normal_mode_displacements_{i}", evec)

            save_results(at, ['energy', 'forces', 'dipole'], output_prefix)

        clean_rundir(rundir, keep_files, __default_keep_files,
                     calculation_succeeded)

        # handling of the scratch path for ExtendedORCA
        if not basin_hopping and "scratch_path" in orca_kwargs.keys() and \
                os.path.isdir(rundir):
            shutil.move(rundir, base_rundir)

    if isinstance(atoms, ase.Atoms):
        return at_list[0]
    else:
        return at_list


class BasinHoppingORCA(Calculator):
    implemented_properties = ['energy', 'forces']

    def __init__(self, atoms=None, n_hop=10, n_run=3, scratch_path=None,
                 n_missing_tolerance=1, energy_tol=0.001,
                 forces_tol=0.05, seed="orca", n_orb=10, max_angle=60.,
                 smearing=5000., maxiter=500, chained_hops=True,
                 orcasimpleinput='UHF PBE def2-SVP tightscf', orcablocks=None,
                 orca_command='orca',
                 keep_files=False, uhf=True, **kwargs):
        """ORCA calculator with basin hopping in wavefunction space for
        smooth PES of radicals

        Method:
        -------
        - call n_runs (3<= recommended) instances of calculation on each
        frame, if all agree in energy within a given
        margin, then it is a successful

        calculation:
        1. no rotation, only smearing calculation
        2. n_hop - 1 number of calculations with random rotations

        Parameters
        ----------
        atoms: ase.Atoms
            molecule for calculation

        n_hop: int
            number of hopping steps

        n_run: int
            number of independent runs to perform, resultant energy/force
            is checked to be the same

        scratch_path: path_like
            put temporary directories here for the calculations, one per task
            very efficient if you have an SSD scratch disk on nodes that
            perform the tasks

        n_missing_tolerance: int
            number of files allowed to be missing

        energy_tol: float
            energy tolerance in eV/atom

        force_tol: float
            force tolerance in eV/Å per component

        seed: str
            file name seed

        n_orb: int
            number of orbital pairs to rotate

        max_angle: float
            maximal angle to rotate orbitals to

        smearing: float, default 5000.
            smearing temperature in K, recommended is 5000 K

        maxiter: int, default 500
            maximum number of SCF iterations to do

        chained_hops: bool, default True
            chain hops, ie. to take the previous wavefunction result or the
            very initial one

        orcasimpleinput: str
            What you'd put after the "!" in an orca input file.

        orcablocks: str
            What you'd put in the "% ... end"-blocks.

        orca_command: str
            command of ORCA calculator as in path

        keep_files: bool
            to keep the resultant files from each calculation

        uhf: bool, default True
            to use Unrestricted HF/KS method. This is advised to be used at
            all times, implemented for development
            purposes.

        kwargs
        """
        super(BasinHoppingORCA, self).__init__(atoms=atoms, **kwargs)

        # calculator settings
        self.seed = seed
        self.smearing = smearing
        self.maxiter = maxiter
        self.orcasimpleinput = orcasimpleinput
        self.orcablocks = (orcablocks if orcablocks is not None else "")
        self.orca_command = orca_command

        # basin hopping settings
        if n_hop < 1:
            raise ValueError(
                "BasinHoppingORCA: n_hop needs to be at least 1!")
        if n_run < 1:
            raise ValueError(
                "BasinHoppingORCA: n_runs needs to be at least 1!")
        self.n_hop = n_hop
        self.n_run = n_run
        self.n_orb = n_orb
        self.max_angle = max_angle
        self.chained_hops = chained_hops
        self.uhf = uhf

        if scratch_path is not None:
            self.scratch_path = scratch_path
        else:
            self.scratch_path = self.directory

        # tolerances
        self.n_missing_tolerance = n_missing_tolerance
        self.energy_tol = energy_tol
        self.forces_tol = forces_tol

        # for keeping the resultant files if needed
        self.keep_files = keep_files
        self.directory = os.path.abspath(self.directory)

    def calculate(self, atoms=None, properties=None,
                  system_changes=all_changes, keep_raw_results=False,
                  verbose=False):
        if properties is None:
            properties = self.implemented_properties

        # needed dry run of the ase calculator and check if calculation is
        # required
        ase.calculators.calculator.Calculator.calculate(self, atoms,
                                                        properties,
                                                        system_changes)
        if not self.calculation_required(self.atoms, properties):
            return

        # carry out n_runs number of independent optimisations
        collective_energy_array = np.zeros(
            shape=(self.n_run, self.n_hop)) + np.inf
        collective_forces_array = np.zeros(
            shape=(self.n_run, self.n_hop, len(self.atoms), 3)) + np.inf

        for i_run in range(self.n_run):
            e, f = self._calc_one_run_with_tempdir(i_run)

            collective_energy_array[i_run] = e
            collective_forces_array[i_run] = f

        if keep_raw_results:
            self.results["raw_energy"] = collective_energy_array.copy()
            self.results["raw_forces"] = collective_forces_array.copy()

        # processing and add to
        energy, forces = self.process_results(collective_energy_array,
                                              collective_forces_array)
        self.results["energy"] = energy
        self.results["forces"] = forces

    def _calc_one_run_with_tempdir(self, i_run):
        """This is one set of basin hopping calculations from scratch,
        with temporary directory management

        Returns
        -------
        energy_array
        forces_array

        """
        base_dir = os.getcwd()
        tmp = self._make_tempdir()
        os.chdir(tmp)

        try:
            energy_array, forces_array = self._calc_one_run()
        except Exception as e:
            print(
                f"Error encountered in job {os.getcwd()} with the "
                f"calculation:")
            traceback.print_exc(e)

            # initialise to correct default
            energy_array = np.zeros(shape=(self.n_hop,)) + np.inf
            forces_array = np.zeros(
                shape=(self.n_hop, len(self.atoms), 3)) + np.inf
        finally:
            self._take_files_after_run(tmp, i_run)
            if os.path.isdir(tmp):
                shutil.rmtree(tmp)

        os.chdir(base_dir)
        return energy_array, forces_array

    def _calc_one_run(self):
        """This is one set of basin hopping calculations from scratch,
        temporary directory not included yet

        Returns
        -------
        energy_array
        forces_array

        """

        energy_array = np.zeros(shape=(self.n_hop,)) + np.inf
        forces_array = np.zeros(
            shape=(self.n_hop, len(self.atoms), 3)) + np.inf

        init = True
        for i_hop in range(self.n_hop):
            try:
                at = self._copy_atoms()
                at.calc = self._generate_new_calculator(init)

                init = False  # do this only once, even if initial
                # calculation fails

                energy_array[i_hop] = at.get_potential_energy()
                forces_array[i_hop] = at.get_forces()
            except Exception as e:
                sys.stderr.write(
                    f"Calculation failed at {i_hop}, continuing -- err is: "
                    f"{e}")
            finally:
                for extension in ["gbw", "inp", "out", "ase", "engrad"]:
                    # saving out, wavefunction and energy results as well
                    try:
                        shutil.copy(f"{self.seed}.{extension}",
                                    f"{self.seed}.{i_hop}.{extension}")  #
                        # saving the output file
                    except FileNotFoundError:
                        pass

                if not self.chained_hops and i_hop != 0:
                    # always use the first calculation's wavefunction as
                    # the starting point
                    if os.path.isfile(f"{self.seed}.0.gbw"):
                        shutil.copy(f"{self.seed}.0.gbw", f"{self.seed}.gbw")
                    else:
                        sys.stderr.write(
                            f"Have not found {self.seed}.0.gbw after hop "
                            f"{i_hop}, "
                            f"so only chained hops are possible")

        return energy_array, forces_array

    def process_results(self, energy_array, force_array):
        """Compares the resultant energies and forces

        Maximal difference between energy minima per atom and corresponding
        force components are checked against the
        set thresholds.

        Parameters
        ----------
        energy_array: array, shape(n_runs, n_hop)
        force_array: array, shape(n_runs, n_hop, len_atoms, 3)

        Returns
        -------
        energy
        forces

        """

        # check sizes
        assert energy_array.shape == (self.n_run, self.n_hop)
        assert force_array.shape == (
            self.n_run, self.n_hop, len(self.atoms), 3)

        # decide if results are acceptable
        energy_min_per_run = np.min(energy_array, axis=1)
        non_inf = energy_min_per_run < np.inf
        num_non_inf = int(np.sum(non_inf))

        if num_non_inf == 0 or num_non_inf < self.n_run - \
                self.n_missing_tolerance:
            # not enough calculations succeeded
            raise CalculationFailed(
                f"Not enough runs succeeded ({num_non_inf}) in wavefunction "
                f"basin hopping")

        # check energy maximal difference
        energy_max_diff = np.max(energy_min_per_run[non_inf]) - np.min(
            energy_min_per_run)
        if energy_max_diff > self.energy_tol * len(self.atoms):
            raise CalculationFailed(
                f"Too high energy difference found: {energy_max_diff} at "
                f"tolerance: "
                f"{self.energy_tol * len(self.atoms)}")

        # take the correct force arrays, the first [] gives a dim4 array
        # with one element along axis0, reduced to dim3
        forces_reduced_arrays = force_array[
            np.where(non_inf), np.argmin(energy_array, axis=1)[non_inf]][0]
        forces_max_diff = np.max(np.abs(
            forces_reduced_arrays[
                np.atleast_1d(np.argmax(energy_min_per_run[non_inf]))[0]] -
            forces_reduced_arrays[
                np.atleast_1d(np.argmin(energy_min_per_run))[0]]
        ))
        if forces_max_diff > self.forces_tol:
            raise CalculationFailed(
                f"Too high force difference found: {forces_max_diff} at "
                f"tolerance: {self.forces_tol}")

        return np.min(energy_min_per_run), forces_reduced_arrays[
            np.atleast_1d(np.argmin(energy_min_per_run))[0]].copy()

    def _take_files_after_run(self, tmp_path, i_run):
        """Takes the desired amount of files from the scratch dirs after run

        In order for ase.calculators.orca.ORCA.read(label) to work,
        the following files are needed:
        - .inp, .out, .ase, .engrad

        Parameters
        ----------
        tmp_path

        """
        if self.keep_files:
            for i_hop in range(self.n_hop):
                for ext in ["inp", "out", "ase", "engrad"]:
                    fn = os.path.join(tmp_path, f"{self.seed}.{i_hop}.{ext}")
                    if os.path.isfile(fn):
                        shutil.move(fn, os.path.join(self.directory,
                                                     f"{self.seed}.run_"
                                                     f"{i_run}.hop_"
                                                     f"{i_hop}.{ext}"))

    def _copy_atoms(self):
        """Copy the internal atoms object, needed for multiprocessing

        Returns
        -------
        atoms: Atoms
            new atoms object, containing elements and positions of the
            internal one

        """
        return ase.Atoms(self.atoms.get_chemical_symbols(),
                         positions=self.atoms.get_positions())

    def _generate_new_calculator(self, initial=False):
        """Creates a single use ORCA calculator with the perturbations

        Parameters
        ----------
        initial : bool
            True  -> No mix, no autostart in case there is a .gbw file in
            the directory
            False -> n_rot number of mixes generated
        """
        if initial:
            rot_string = "AutoStart false"
        else:
            rot_string = self._generate_perturbations()

        calc = ExtendedORCA(
            label=self.seed,
            orca_command=self.orca_command,
            charge=0, mult=self.get_multiplicity(),
            task='engrad',
            orcasimpleinput=self.orcasimpleinput,
            orcablocks=f"%scf SmearTemp {self.smearing} \nmaxiter "
                       f"{self.maxiter} \n"
                       f"  {rot_string} \nend \n{self.orcablocks}"
        )
        return calc

    def _generate_perturbations(self):
        """ Generates the perturbations for the calculations

        Notes
        -----
        This may fail for small systems or small basis sizes,
        where the number of created virtual orbitals is not enough.
        """
        # HOMO indices, 0-based
        i_homo_a, i_homo_b = self.get_homo()

        # create the indices of the occ/virtual orbs to mix
        idx_occ_a = np.arange(max(0, i_homo_a + 1 - self.n_orb), i_homo_a + 1)
        idx_occ_b = np.arange(max(0, i_homo_b + 1 - self.n_orb), i_homo_b + 1)

        idx_virtual_a = np.arange(i_homo_a, i_homo_a + self.n_orb) + 1
        idx_virtual_b = np.arange(i_homo_b, i_homo_b + self.n_orb) + 1

        # reset the number of rotations there is more than the 2/3 number
        # of electrons in the system
        n_from_a = min(np.random.randint(self.n_orb + 1),
                       int(i_homo_a * 2 / 3) + 1)
        n_from_b = min(self.n_orb - n_from_a, int(i_homo_b * 2 / 3) + 1)

        n_rotations_used = n_from_a + n_from_b
        idx_occ = np.concatenate(
            [np.random.choice(idx_occ_a, n_from_a, replace=False),
             np.random.choice(idx_occ_b, n_from_b, replace=False)])
        idx_virtual = np.concatenate(
            [np.random.choice(idx_virtual_a, n_from_a, replace=False),
             np.random.choice(idx_virtual_b, n_from_b, replace=False)])
        spin = np.zeros(n_rotations_used, dtype=int)
        spin[n_from_a:] += 1

        angles = np.random.rand(n_rotations_used) * self.max_angle

        rot_string = "Rotate \n"
        for i in range(n_rotations_used):
            if self.uhf or spin[i] == 0:
                # in RHF we can only mix the alpha spins
                rot_string += f"  {{{idx_occ[i]:3}, {idx_virtual[i]:3}, " \
                              f"{angles[i]:10.4}, {spin[i]}, {spin[i]}}} \n"
        rot_string += "end \n"

        return rot_string

    def get_homo(self):
        """ Get the index of the HOMO.

        ORCA counts form 0 and fills the alpha (indexed 0) spin more than
        the beta (indexed 1) for odd n_elec.

        Returns
        -------
        i_homo_alpha, i_homo_beta : int
        """
        n_elec = np.sum(self.atoms.get_atomic_numbers())
        if n_elec % 2 == 0:
            val = n_elec // 2 - 1
            return val, val
        else:
            return n_elec // 2, n_elec // 2 - 1

    def get_multiplicity(self):
        """ Gets the multiplicity of the atoms object.

        ORCA format: singlet = 1, doublet = 2, etc.

        Returns
        -------
        multiplicity : int
        """

        return ExtendedORCA.get_default_multiplicity(self.atoms)

    def _make_tempdir(self, prefix="orca_", suffix=None):
        # makes a temporary directory in at the scratch path set
        # NOTE: this needs to be deleted afterwards!!!
        return tempfile.mkdtemp(dir=self.scratch_path, prefix=prefix,
                                suffix=suffix)


class ExtendedORCA(ORCA):
    """Extension of ASE's ORCA calculator with the following features:
        - specify command for executable (ase devs don't let this to be
        merged in)
        - setting multiplicity:
            default in the properties in None, which triggers the
            calculator to set to singlet or doublet
        - geometry optimisation
        - frequencies

    Handling of "task":
    1. directly written to file
    2. default is "engrad", which is appended if not in task str
    3. None -> "engrad"

    For geometry optimisation and freq on the last frame, use task="opt freq".
    """

    implemented_properties = ["energy", "forces", "dipole"]

    # same as parent class, only multiplicity changed to trigger default
    default_parameters = dict(
        charge=0, mult=None,
        task='engrad',
        orcasimpleinput='tightscf PBE def2-SVP',
        orcablocks='%scf maxiter 200 end')

    def __init__(self, restart=None,
                 ignore_bad_restart_file=Calculator._deprecated,
                 label='orca', atoms=None, **kwargs):
        super(ExtendedORCA, self).__init__(restart, ignore_bad_restart_file,
                                           label, atoms, **kwargs)
        # this is missing from the ase calculator and Ask was stubborn not
        # to include it in a PR, shame
        if 'orca_command' in kwargs:
            self.command = f'{str(kwargs.get("orca_command"))} PREFIX.inp > ' \
                           f'PREFIX.out'

        self.extra_results = dict()

    def write_input(self, atoms, properties=None, system_changes=None):
        """This is setting the multiplicity, unless set previously

        Adapted from ase.io.orca.write_orca()

        Extensions:
            - choice of task
            - task parameter is directly inserted into the file
        """

        if self.parameters.get("mult", None) is None:
            self.set(mult=self.get_default_multiplicity(atoms,
                                                        self.parameters.get(
                                                            "charge", 0)))

        # copy of ASE's method, just using the patched write_orca() from here
        FileIOCalculator.write_input(self, atoms, properties, system_changes)
        self.parameters.write(self.label + '.ase')

        # this is modified in place here
        orcablocks = self.parameters['orcablocks']

        if self.pcpot:
            pcstring = f'% pointcharges \"{self.label}.pc\"\n\n'
            orcablocks += pcstring
            self.pcpot.write_mmcharges(self.label)

        with open(self.label + '.inp', 'w') as f:
            # energy and force calculation is enforced
            task = self.parameters["task"]
            if task is None:
                task = "engrad"
            elif "engrad" not in task and "opt" not in task:
                task += " engrad"

            f.write(f"! {task} {self.parameters['orcasimpleinput']} \n")
            f.write(f"{orcablocks} \n")

            f.write('*xyz')
            f.write(f" {self.parameters['charge']:d}")
            f.write(f" {self.parameters['mult']:d} \n")
            for atom in atoms:
                if atom.tag == 71:  # 71 is ascii G (Ghost)
                    symbol = atom.symbol + ' : '
                else:
                    symbol = atom.symbol + '   '
                f.write(symbol +
                        str(atom.position[0]) + ' ' +
                        str(atom.position[1]) + ' ' +
                        str(atom.position[2]) + '\n')
            f.write('*\n')

    def is_converged(self):
        """checks for warnings about SCF/wavefunction not converging.

        Returns
            None if "FINAL SINGLE POINT ENERGY" is not in the output
            False if "Wavefunction not fully converged" is in the file
            True otherwise

        Based on ase.calculators.orca.read_energy().

        """
        with open(self.label + '.out', mode='r', encoding='utf-8') as fd:
            text = fd.read()

        re_energy = re.compile(r"FINAL SINGLE POINT ENERGY.*\n")
        re_not_converged = re.compile(r"Wavefunction not fully converged")
        found_line = re_energy.search(text)

        if found_line is None:
            return None

        return not re_not_converged.search(found_line.group(0))

    def read_results(self):
        """Reads all results"""

        if not self.is_converged():
            raise CalculationFailed("Wavefunction not fully converged")

        self.read_energy()
        self.read_forces()

        self.read_dipole()
        if 'opt' in self.parameters.task:
            self.read_opt_atoms()
            self.read_trajectory()
        if 'freq' in self.parameters.task:
            self.read_frequencies()

    def read_opt_atoms(self):
        """Reads the result of the geometry optimisation"""
        opt_atoms = ase.io.read(f'{self.label}.xyz')
        self.extra_results["relaxed_positions"] = opt_atoms.positions.copy()

    def read_trajectory(self):
        """Reads the trajectory of the geometry optimisation

        copied from normal_modes branch, code by eg475

        Notes
        -----
        Each comment line in output xyz has "Coordinates from ORCA-job orca
        E -154.812399026326";
        tks32:
        However the forces are calculated at each step as well, so that is
        in the output file,
        meaning we can parse it from there if we actually need it,
        which can let us use the energy
        from there as well.
        """
        opt_trj = ase.io.read(f'{self.label}_trj.xyz', ':')
        for at in opt_trj:
            energy = None
            for key in at.info.keys():
                try:
                    energy = float(key) * units.Ha
                except ValueError:
                    continue

            at.info.clear()
            if energy is not None:
                at.info['energy'] = energy

        self.extra_results["opt_trajectory"] = opt_trj

    def read_frequencies(self):
        with open(self.label + '.out', mode='r', encoding='utf-8') as fd:
            text = fd.read()

        # ASE craziness: the self.atoms attribute can be forgotten within
        # the calculation ;)
        if self.atoms is not None:
            len_atoms = len(self.atoms)
        else:
            self.read_forces()
            len_atoms = len(self.results["forces"])
        n_free = 3 * len_atoms

        # patterns used here
        pattern_eig_block = re.compile(
            r"VIBRATIONAL FREQUENCIES\n[-]+\n\s?\nSc.*\n\s?\n(("
            r"?:\s+\d+:\s+-?\d+\.\d+\scm"
            r"\*\*-1\s?(?:\s\*\*\*imaginary\smode\*\*\*)?\n)+)")
        pattern_eig_line = re.compile(
            r"(?P<idx>\d+):\s+(?P<value>-?\d+\.\d+)\scm\*\*-1")
        pattern_nm_block = re.compile(
            r"NORMAL MODES\n[-]+\n\s?\n(?:.*\n){3}\s?\n([-\s\d.]+)\n\n")
        pattern_nm_line = re.compile(r"\s+\d+((?:\s+-?\d\.\d+)+)")

        # eigenvalues
        frequency_block = pattern_eig_block.findall(text)
        if frequency_block:
            # read the frequencies
            vib_energies = np.zeros(shape=n_free, dtype=float)
            for line in frequency_block[0].split("\n"):
                line_match = pattern_eig_line.search(line)
                if line_match:
                    idx, value = line_match.groups()
                    vib_energies[int(idx)] = float(value) * units.invcm

            self.extra_results["normal_mode_eigenvalues"] = vib_energies
        else:
            raise CalculationFailed(
                "Frequency calculation has failed in ORCA: eigenvalues not "
                "found")

        # eigenvectors modes
        vectors_block = pattern_nm_block.findall(text)
        if vectors_block:
            # take only the vectors from lines
            number_lines = []
            for line in vectors_block[0].split("\n"):
                mm = pattern_nm_line.search(line)
                if mm:
                    number_lines.append(mm.groups()[0])

            # read & reshape to square array
            nm_eigenvectors = np.zeros(shape=(n_free, n_free), dtype=float)
            i_mode = 0
            for chunk in chunks(number_lines, n_free):
                num_new_modes = len(chunk[0].split())
                for i_coord, line in enumerate(chunk):
                    numbers = np.array([float(val) for val in line.split()])
                    nm_eigenvectors[i_mode:i_mode + num_new_modes,
                    i_coord] = numbers

                i_mode += num_new_modes

            # fill in results, shape: (3N, N, 3)
            self.extra_results["normal_mode_displacements"] = np.reshape(
                nm_eigenvectors,
                newshape=(n_free, len_atoms, 3))
        else:
            raise CalculationFailed(
                "Frequency calculation has failed in ORCA: eigenvectors not "
                "found")

    def read_dipole(self):
        """
        Dipole is calculated by default, though only written up to 0.00001
        so the rest of
        the digits are not meaningful in the output.
        """
        with open(self.label + '.out', mode='r', encoding='utf-8') as fd:
            text = fd.read()

        # recognise the whole block and pick up the three numbers we need
        pattern_dipole_block = re.compile(
            r"(?:DIPOLE MOMENT\n"
            r"[-]+\n"
            r"[\sXYZ]*\n"
            r"Electronic contribution:[\s\.0-9-]*\n"
            r"Nuclear contribution   :[\s\.0-9-]*\n"
            r"[-\s]+\n"
            r"Total Dipole Moment    :\s+([-0-9\.]*)\s+([-0-9\.]*)\s+([-0-9\.]*))"
            # this reads the X, Y, Z parts
        )

        # three numbers, total dipole in a.u.
        match = pattern_dipole_block.search(text)

        if match:
            dipole = np.array(
                [float(x) for x in match.groups()]) / units.Debye * units.Bohr
            self.results["dipole"] = dipole

    @staticmethod
    def get_default_multiplicity(atoms, charge=0):
        """ Gets the multiplicity of the atoms object.

        ORCA format: singlet = 1, doublet = 2, etc.

        Parameters
        ----------
        atoms : ase.Atoms
        charge : int

        Returns
        -------
        multiplicity : int
        """

        return np.sum(atoms.get_atomic_numbers() - int(charge)) % 2 + 1
