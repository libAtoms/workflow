import os
import pathlib
import re
import shutil
import sys
import tempfile
import traceback
import warnings

import ase.io
import numpy as np
from ase import units
from ase.calculators.calculator import CalculationFailed, Calculator, \
    FileIOCalculator, all_changes
from ase.calculators.orca import ORCA

from wfl.calculators.utils import clean_rundir, save_results
from wfl.pipeline import iterable_loop
from wfl.utils.misc import atoms_to_list, chunks
from  wfl.calculators.orca.bh import BasinHoppingORCA

__default_keep_files = ["*.inp", "*.out", "*.ase", "*.engrad", "*.xyz",
                        "*_trj.xyz"]


def evaluate(inputs, outputs,
             base_rundir=None, dir_prefix="ORCA_", keep_files="default",
             orca_kwargs=None,
             output_prefix=None, basin_hopping=False):
    """Evaluate with ORCA calculator, optionally BasinHopping

    Parameters
    ----------
    inputs: list(Atoms) / Configset_in
        input atomic configs, needs to be iterable
    outputs: list(Atoms) / Configset_out
        output atomic configs
    base_rundir: path-like, default os.getcwd()
        directory to put calculation directories into
    dir_prefix: str, default 'ORCA\_'
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
    dir_prefix: str, default 'ORCA\_'
        directory name prefix for calculations
    keep_files: "default" / bool
        what kind of files to keep from the run
            - "default : .out, .inp, .ase, .engrad is kept -- ase can read
              the results again
            - True : all files kept
            - False: none kept
    orca_kwargs: dict
        kwargs for BasinHoppingORCA calculator
    output_prefix : str, default "ORCA\_"
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
            # ORCA has no concept of scratch_path, but setting it
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
            at.calc = ORCA(**dict(orca_kwargs, directory=rundir))

        # skip if calculation fails
        calculation_succeeded = False
        try:
            at.calc.calculate(at)
            calculation_succeeded = True
            if 'DFT_FAILED_ORCA' in at.info:
                del at.info['DFT_FAILED_ORCA']
        except Exception as exc:
            warnings.warn(f'Calculation failed with exc {exc}')
            at.info['DFT_FAILED_ORCA'] = True

        if calculation_succeeded and not basin_hopping:
            # task='opt' in ORCA performs geometry optimisation,
            # results are for the relaxed positions
            # alternative option: task='copt' for cartesian optimisation
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

        # handling of the scratch path for ORCA
        if not basin_hopping and "scratch_path" in orca_kwargs.keys() and \
                os.path.isdir(rundir):
            shutil.move(rundir, base_rundir)

    if isinstance(atoms, ase.Atoms):
        return at_list[0]
    else:
        return at_list


class ORCA(ORCA):
    """Extension of ASE's ORCA calculator with the following features:

        - specify command for executable (ase devs don't let this to be
          merged in)
        - setting multiplicity: default in the properties in None, \
            which triggers the calculator to set to singlet or doublet
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
        super(ORCA, self).__init__(restart, ignore_bad_restart_file,
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
            elif "engrad" not in task and "opt" not in task and "copt" not in task:
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
            ``None`` if "FINAL SINGLE POINT ENERGY" is not in the output

            ``False`` if "Wavefunction not fully converged" is in the file

            ``True`` otherwise

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
        if 'opt' in self.parameters.task or "copt" in self.parameters.task:
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
        Note
        ----

        Dipole is calculated by default, though only written up to 0.00001
        so the rest of  the digits are not meaningful in the output.
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
