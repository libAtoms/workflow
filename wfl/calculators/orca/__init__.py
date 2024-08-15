import re
from pathlib import Path
import subprocess

import ase.io
from ase.io.orca import read_geom_orcainp
import numpy as np
from ase import units
from ase.calculators.calculator import CalculationFailed, all_changes 
from ase.calculators.orca import ORCA as ASE_ORCA

from ..wfl_fileio_calculator import WFLFileIOCalculator
from wfl.utils.misc import chunks


_default_keep_files = ["*.inp", "*.out",  "*.engrad", "*.xyz",
                        "*_trj.xyz"]
_default_properties = ["energy", "forces"]


class ORCA(WFLFileIOCalculator, ASE_ORCA):
    """Extension of ASE's ORCA calculator that can be used by wfl.calculators.generic

    Notes
    -----
        - "directory" argument cannot be present. Use rundir_prefix and workdir instead.
        - Unless specified, multiplicity is set to singlet/doublet for
          closed-/open-shell structures
        - Results from additional optimisation task (set to "opt" or "copt") are stored in `extra_results` dictionary.

    Parameters
    ----------
    keep_files: bool / None / "default" / list(str), default "default"
        what kind of files to keep from the run
            - True : everything kept
            - None, False : nothing kept, unless calculation fails
            - "default"   : only ones needed for NOMAD uploads ('\*.pwo')
            - list(str)   : list of file globs to save
    rundir_prefix: str / Path, default 'ORCA\_'
        Run directory name prefix
    workdir: str / Path, default . at calculate time
        Path in which rundir will be created.
    scratchdir: str / Path, default None
        temporary directory to execute calculations in and delete or copy back results (set by
        "keep_files") if needed.  For example, directory on a local disk with fast file I/O.
    post_process: function that takes the current instance of the calculator and is to be
        executed after reading back results from file, but before all the files are deleted.
        For example, a localisation scheme that uses the wavefunction files and saves local
        charges to `ORCA.extra_results`.

    **kwargs: arguments for ase.calculators.orca.ORCA
        see https://wiki.fysik.dtu.dk/ase/ase/calculators/orca.html#module-ase.calculators.orca
    """

    implemented_properties = ["energy", "forces", "dipole"]

    # new default value of num_inputs_per_python_subprocess for calculators.generic,
    # to override that function's built-in default of 10
    wfl_generic_default_autopara_info = {"num_inputs_per_python_subprocess": 1}

    default_params = dict(charge=0, orcasimpleinput='engrad B3LYP def2-TZVP',
                  orcablocks='%pal nprocs 1 end')


    def __init__(self, keep_files="default", rundir_prefix="ORCA_", scratchdir=None,
                 workdir=None, post_process=None, **kwargs):

        super().__init__(keep_files=keep_files, rundir_prefix=rundir_prefix,
                         workdir=workdir, scratchdir=scratchdir, **kwargs)

        # to make use of wfl.utils.save_calc_results.save_calc_results()
        self.extra_results = dict()
        self.extra_results["atoms"] = {}
        self.extra_results["config"] = {}

        self.post_process = post_process


    def calculate(self, atoms=None, properties=_default_properties, system_changes=all_changes):
        """Does the calculation. Handles the working directories in addition to regular
        ASE calculation operations (writing input, executing, reading_results) """

        # should be fixed by ASE's PR #3442
        if self.atoms is None:
            self.atoms = atoms

        # from WFLFileIOCalculator
        self.setup_rundir()

        self.fill_in_default_params()

        self.enforce_force_calculation()

        if self.parameters.get("mult", None) is None:
            charge = self.parameters.get("charge", 0)
            self.parameters["mult"] = self.get_default_multiplicity(atoms, charge)

        try:
            super().calculate(atoms=atoms, properties=properties, system_changes=system_changes)

            self.wfl_calc_post_process()

            calculation_succeeded = True
            if 'DFT_FAILED_ORCA' in atoms.info:
                del atoms.info["DFT_FAILED_ORCA"]
        except Exception as e:
            atoms.info["DFT_FAILED_ORCA"] = True
            calculation_succeeded = False
            raise e
        finally:
            self.clean_rundir(_default_keep_files, calculation_succeeded)

    def wfl_calc_post_process(self): 
        """
        Extends ASE's calculator's reading capabilities. 
        - Checks the calculation has converged and raises an exception if not
        - energy, force and dipole are read by ASE
        - reads optimised aptoms and optimisation trjectory, if appropriate. 
        - (to be updated) reads frequencies, and eigenmodes, if appropriate. 
        - performs post-procesing with any external functions
        """

        output_file_path = self.directory / self.template.outputname

        if not self.is_converged(output_file_path):
            raise CalculationFailed("Wavefunction not fully converged")

        orcasimpleinput = self.parameters["orcasimpleinput"]

        if 'opt' in orcasimpleinput or "copt" in orcasimpleinput:
            self.read_opt_atoms(str(output_file_path).replace(".out", ".xyz"))
            self.read_trajectory(str(output_file_path).replace(".out", "_trj.xyz"))

        # to be updated
        #if 'freq' in orcasimpleinput:
        #    self.read_frequencies()
        
        if self.post_process is not None:
            self.post_process(self)

    def fill_in_default_params(self):

        parameters = self.default_params.copy()
        parameters.update(self.parameters)
        self.parameters = parameters


    def enforce_force_calculation(self):
        
        orcasimpleinput = self.parameters["orcasimpleinput"]

        # does "freq" produce engrad too?
        force_producing_keys = ["engrad", "opt", "copt"]
        expect_engrad = np.sum([key in orcasimpleinput for key in force_producing_keys])

        if not expect_engrad:
            self.parameters["orcasimpleinput"] = "engrad " + orcasimpleinput


    def is_converged(self, output_file_path):
        """checks for warnings about SCF/wavefunction not converging.

        Returns
            ``None`` if "FINAL SINGLE POINT ENERGY" is not in the output
            ``False`` if "Wavefunction not fully converged" is in the file
            ``True`` otherwise

        Based on ase.calculators.orca.read_energy().

        """

        with open(output_file_path, mode='r', encoding='utf-8') as fd:
            text = fd.read()

        re_energy = re.compile(r"FINAL SINGLE POINT ENERGY.*\n")
        re_not_converged = re.compile(r"Wavefunction not fully converged")
        found_line = re_energy.search(text)

        if found_line is None:
            return None

        return not re_not_converged.search(found_line.group(0))


    def read_opt_atoms(self, out_xyz_fn):
        """Reads the result of the geometry optimisation"""
        opt_atoms = ase.io.read(out_xyz_fn)
        self.extra_results["relaxed_positions"] = opt_atoms.positions.copy()


    def read_trajectory(self, traj_fn):
        """Reads the trajectory of the geometry optimisation

        Notes
        -----
        Each comment line in output xyz has "Coordinates from ORCA-job orca
        E -154.812399026326";
        # TODO parse out forces as well
        """
        opt_trj = ase.io.read(traj_fn, ":")
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

    orca_input_fn = orca_calc.directory / orca_calc.template.inputname
    label = orca_input_fn.parent / orca_input_fn.stem

    # just to get the elements
    if orca_calc.atoms is not None:
        ref_elements = list(orca_calc.atoms.symbols)
    else:
        atoms = read_geom_orcainp(orca_input_fn)
        ref_elements = list(atoms.symbols)


    # 1. Convert from orca output to incomplete molden
    orca_command = orca_calc.profile.command
    command = f"{orca_command}_2mkl {label} -molden > {label}.orca_2mkl.out"
    subprocess.run(command, shell=True)     # think about how to handle errors, etc

    # 2. Clean up molden format
    command = (f"java -jar {janpa_home_dir / 'molden2molden.jar'} -i {label}.molden.input -o {label}.molden.cleanedup "
               f"-fromorca3bf -orca3signs > {label}.molden2molden.stdout")
    subprocess.run(command, shell=True)

    # 3. Run natural population analysis
    npa_output = f"{label}.janpa"
    command = f'java -jar {janpa_home_dir / "janpa.jar"} -i {label}.molden.cleanedup > {npa_output}'
    subprocess.run(command, shell=True)

    # 4. Save results
    elements, electron_pop, npa_charge = parse_npa_output(npa_output)
    assert np.all([v1 == v2 for v1, v2 in zip(elements, ref_elements)])
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

    block = pattern_npa_block.findall(text)[0]
    for line in block.split('\n'):
        values = pattern_entry.search(line)
        if values is not None:
            values = values.groups()
            elements.append(values[0])
            electron_pop.append(float(values[1]))
            npa_charge.append(float(values[2]))

    return elements, np.array(electron_pop), np.array(npa_charge)
