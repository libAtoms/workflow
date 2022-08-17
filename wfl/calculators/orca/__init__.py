import os
import re
import shutil
import tempfile
import warnings
from pathlib import Path 
import subprocess

import ase.io
from ase.io.orca import read_geom_orcainp
import numpy as np
from ase import units
from ase.calculators.calculator import CalculationFailed, Calculator, \
    FileIOCalculator, all_changes
import ase.calculators.orca

from wfl.calculators.utils import clean_rundir, save_results
from wfl.utils.misc import atoms_to_list, chunks
from wfl.calculators.orca.basinhopping import BasinHoppingORCA

default_keep_files = ["*.inp", "*.out", "*.ase", "*.engrad", "*.xyz",
                        "*_trj.xyz"]


class ORCA(ase.calculators.orca.ORCA):
    """Extension of ASE's ORCA calculator:
        - Specify Path to orca executable
        - Unless specified, multiplicity is set to singlet/doublet for
          closed-/open-shell structures
        - Additional tasks with results stored in `extra_results` dictionary. 

    Handling of "task":
        - ORCA programs to run, in addition or instead of calculating energy forces and dipole
        - `task` will be pre-pended to the keyword line
        - `engrad` is enforced by default, except in the case of geometry optimisation
        - Supported conventional tasks: `opt`, `copt` (see orca manual)

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
                 label='orca', atoms=None,
                 keep_files="default", dir_prefix="ORCA_", scratch_path=None,
                 orca_path="orca", post_process=None,
                 **kwargs):

        super(ORCA, self).__init__(restart, ignore_bad_restart_file,
                                           label, atoms, **kwargs)

        self.orca_path = orca_path
        if self.orca_path != "orca":
            self.command = f'{orca_path} PREFIX.inp > ' \
                           f'PREFIX.out'

        self.extra_results = dict()
        # to make use of wfl.calculators.utils.save_results()
        self.extra_results["atoms"] = {}
        self.extra_results["config"] = {}


        self.keep_files = keep_files
        # set up the directories
        self.scratch_path = scratch_path
        # self.directory is overwritten in self.calculate, so let's just keep track
        self.dir_prefix = dir_prefix
        self.workdir_root = Path(self.directory) / (self.dir_prefix + "calc_files")
        self.workdir_root.mkdir(parents=True, exist_ok=True)

        self.post_process = post_process 


    def calculate(self, atoms=None, properties=["energy", "forces"], system_changes=all_changes):
        """Does the calculation. Handles the working directories in addition to regular 
        ASE calculation operations (writing input, executing, reading_results) """

        Calculator.calculate(self, atoms, properties, system_changes)

        if self.scratch_path is not None:
            self.directory = tempfile.mkdtemp(dir=self.scratch_path, prefix=self.dir_prefix)
        else:
            self.directory = tempfile.mkdtemp(dir=self.workdir_root, prefix=self.dir_prefix)

        try:
            self.write_input(self.atoms, properties, system_changes)
            self.execute()
            self.read_results()
            if self.post_process is not None:
                self.post_process(self)
            calculation_succeeded=True
        except Exception as e:
            calculation_succeeded=False
            raise e
        finally:
            # when exception is raised, `calculation_succeeded` is set to False, 
            # the following code is executed and exception is re-raised. 
            clean_rundir(self.directory, self.keep_files, default_keep_files, calculation_succeeded)
            if self.scratch_path is not None and Path(self.directory).exists():
                shutil.move(self.directory, self.workdir_root)


    def cleanup(self):
        """Clean all (empty) directories that could not have been removed
        immediately after the calculation, for example, because other parallel
        process might be using them."""
        if any(self.workdir_root.iterdir()):
            print(f'{self.workdir_root.name} is not empty, not removing')
        else:
            self.workdir_root.rmdir()


    def write_input(self, atoms, properties=None, system_changes=None):
        """Writes orca.inp, based on the wfl ORCA calculator parameters"""

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

            f.write(f"! {self.pick_task()} {self.parameters['orcasimpleinput']} \n")
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
    
    def pick_task(self):
        # energy and force calculation is enforced
        task = self.parameters["task"]
        if task is None:
            task = "engrad"
        elif "engrad" not in task and "opt" not in task and "copt" not in task:
            task += " engrad"
        return task

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

        Notes
        -----
        Each comment line in output xyz has "Coordinates from ORCA-job orca
        E -154.812399026326";
        # TODO parse out forces as well 
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

    # EG: The eigenvector reading is implemented incorrectly 
    def read_frequencies(self):
        """Reads frequencies (dynamical matrix eigenvalues) and normal modes (eigenvectors) from output. 
        Currently broken. """
        raise NotImplementedError("Normal mode (eigenvector) parsing is broken")
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


def natural_population_analysis(janpa_home_dir, orca_calc):
    """https://sourceforge.net/p/janpa/wiki/Home/"""

    janpa_home_dir = Path(janpa_home_dir)

    label = orca_calc.label

    # just to get the elements
    if orca_calc.atoms is not None:
        ref_elements = list(orca_calc.atoms.symbols)
    else:
        atoms = read_geom_orcainp(label + '.inp')
        ref_elements = list(atoms.symbols)


    # 1. Convert from orca output to incomplete molden
    command = f"{orca_calc.orca_path}_2mkl {label} -molden > {label}.orca_2mkl.out"
    subprocess.run(command, shell=True)     # think about how to handle errors, etc

    # 2. Clean up molden format
    command = f"java -jar {janpa_home_dir / 'molden2molden.jar'} -i {label}.molden.input -o {label}.molden.cleanedup -fromorca3bf -orca3signs > {label}.molden2molden.stdout"
    subprocess.run(command, shell=True)

    # 3. Run natural population analysis
    npa_output = f"{label}.janpa"
    command = f'java -jar {janpa_home_dir / "janpa.jar"} -i {label}.molden.cleanedup > {npa_output}'
    subprocess.run(command, shell=True)

    # 4. Save results 
    elements, electron_pop, npa_charge = parse_npa_output(npa_output)
    assert np.all([v1==v2 for v1, v2 in zip(elements, ref_elements)])
    orca_calc.extra_results["atoms"]["NPA_electron_population"] = electron_pop 
    orca_calc.extra_results["atoms"]["NPA_charge"] = npa_charge


def parse_npa_output(fname):
    with open(fname, mode='r') as f:
        text = f.read()

    pattern_npa_block = re.compile(
        r"Final electron populations and NPA charges:\n\n"
        r"(?:.*\n)+\nAngular momentum contributions of the total atomic population:")
    pattern_entry = re.compile(r"\s([a-zA-Z]+)(?:\d+)\s+(?:[\d\.]+)\s+([\d\.]+)\s+(?:[\d\.]+)\s+([-\d\.]+)")

    elements = []
    electron_pop = []
    npa_charge = [] 

    # import pdb; pdb.set_trace()
    
    block = pattern_npa_block.findall(text)[0]
    for line in block.split('\n'):
        values = pattern_entry.search(line)
        if values is not None:
            values = values.groups()
            elements.append(values[0])
            electron_pop.append(float(values[1]))
            npa_charge.append(float(values[2]))

    return elements, np.array(electron_pop), np.array(npa_charge)



