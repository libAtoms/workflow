import os
import traceback
import numpy as np
import shutil
import sys
import tempfile

from ase import Atoms
from ase.calculators.calculator import all_changes, CalculationFailed
from ase.calculators.calculator import Calculator

from wfl.calculators import orca
# from wfl.autoparallelize import autoparallelize, autoparallelize_docstring


def evaluate_basin_hopping(*args, **kwargs):
    """Evaluate with BasinHoppingORCA calculator

    Parameters
    ----------
    inputs: list(Atoms) / ConfigSet
        input atomic configs, needs to be iterable
    outputs: OutputSpec
        output atomic configs
    workdir_root: path-like, default os.getcwd()
        directory to put calculation directories into
    rundir_prefix: str, default 'ORCA\_'
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
    results : ConfigSet
        ConfigSet(outputs)
    """
    raise RuntimeError("implemented in terms of wfl.calculators.orca.evaluate_op, which does not exist")
    # return autoparallelize(orca.evaluate_op, *args, basin_hopping=True, **kwargs)
# evaluate_basin_hopping.__doc__ = autoparallelize_docstring(orca.evaluate_op, "Atoms")


class BasinHoppingORCA(Calculator):
    """ORCA calculator with basin hopping in wavefunction space for
    smooth PES of radicals

    Call n_runs (3 <= recommended) instances of calculation on each
    frame. If all agree in energy within a given margin, then it is a successful.

    calculations:

    1. no rotation, only smearing calculation
    2. `n_hop - 1` number of calculations with random rotations

    Parameters
    ----------
    atoms: ase.Atoms
        molecule for calculation

    n_hop: int
        number of hopping steps

    n_run: int
        number of independent runs to perform, resultant energy/force
        is checked to be the same

    scratchdir: path_like
        put temporary directories here for the calculations, one per task
        very efficient if you have an SSD scratch disk on nodes that
        perform the tasks

    n_missing_tolerance: int
        number of files allowed to be missing

    energy_tol: float
        energy tolerance in eV/atom

    force_tol: float
        force tolerance in eV/Angstrom per component

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

    rng: np.random.Generator
        random number generator

    keep_files: bool
        to keep the resultant files from each calculation

    uhf: bool, default True
        to use Unrestricted HF/KS method. This is advised to be used at
        all times, implemented for development
        purposes.

    kwargs
    """
    implemented_properties = ['energy', 'forces']



    def __init__(self, atoms=None, n_hop=10, n_run=3, scratchdir=None,
                 n_missing_tolerance=1, energy_tol=0.001,
                 forces_tol=0.05, seed="orca", n_orb=10, max_angle=60.,
                 smearing=5000., maxiter=500, chained_hops=True,
                 orcasimpleinput='UHF PBE def2-SVP tightscf', orcablocks=None,
                 orca_command='orca', rng=None,
                 keep_files=False, uhf=True, **kwargs):

        super(BasinHoppingORCA, self).__init__(atoms=atoms, **kwargs)

        assert rng is not None, "Random number generator rng is required"

        # calculator settings
        self.seed = seed
        self.smearing = smearing
        self.maxiter = maxiter
        self.orcasimpleinput = orcasimpleinput
        self.orcablocks = (orcablocks if orcablocks is not None else "")
        self.orca_command = orca_command
        self.rng = rng

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

        if scratchdir is not None:
            self.scratchdir = scratchdir
        else:
            self.scratchdir = self.directory

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
        Calculator.calculate(self, atoms,
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
        return Atoms(self.atoms.get_chemical_symbols(),
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

        calc = orca.ORCA(
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
        # Now BasinHoppingOrca uses a passed-in np.random.Generator, hopefully
        #   the following warnings is indeed no longer relevant
        # warnings.warn("BasinHoppingOrca using np.random, not reproducible")
        n_from_a = min(self.rng.integers(self.n_orb + 1),
                       int(i_homo_a * 2 / 3) + 1)
        n_from_b = min(self.n_orb - n_from_a, int(i_homo_b * 2 / 3) + 1)

        n_rotations_used = n_from_a + n_from_b
        idx_occ = np.concatenate(
            [self.rng.choice(idx_occ_a, n_from_a, replace=False),
             self.rng.choice(idx_occ_b, n_from_b, replace=False)])
        idx_virtual = np.concatenate(
            [self.rng.choice(idx_virtual_a, n_from_a, replace=False),
             self.rng.choice(idx_virtual_b, n_from_b, replace=False)])
        spin = np.zeros(n_rotations_used, dtype=int)
        spin[n_from_a:] += 1

        angles = self.rng.uniform(size=n_rotations_used) * self.max_angle

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

        return orca.ORCA.get_default_multiplicity(self.atoms)

    def _make_tempdir(self, prefix="orca_", suffix=None):
        # makes a temporary directory in at the scratch path set
        # NOTE: this needs to be deleted afterwards!!!
        return tempfile.mkdtemp(dir=self.scratchdir, prefix=prefix,
                                suffix=suffix)
