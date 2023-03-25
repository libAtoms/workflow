"""
MOPAC interface
"""
import re
import os
import numpy as np
from packaging import version

from ase.units import kcal,  mol
from ase.calculators.calculator import CalculationFailed, Calculator, \
    FileIOCalculator, all_changes

from .wfl_fileio_calculator import WFLFileIOCalculator

_default_keep_files = ["*.out"]
_default_properties = ["energy", "forces"] 

class MOPAC(WFLFileIOCalculator, FileIOCalculator):
    """ Extension of ASE's MOPAC claculator so that it can be used by wfl.calculators.generic (mainly each calculation is run in a separate directory)"""

    implemented_properties = ["energy", "forces"]

    default_parameters = dict(
        task = "AM1 GRADIENTS RELSCF=0.0001",
        charge = 0,
        mult = None,
    )

    mult_keywords = {
        2: "DOUBLET",
        3: "TRIPLET",
        4: "QUARTET",
        5: "QUINTET",
    }

    wfl_generic_def_autopara_info = {"num_inputs_per_python_subprocess": 1}

    def __init__(self, keep_files="default", rundir_prefix="MOPAC_",
                 workdir=None, scratchdir=None,
                 calculator_exec=None, restart=None, 
                 ignore_bad_restart_file = FileIOCalculator._depreciated,
                 label="mopac", atoms=None, **kwargs):

        if calculator_exec is not None:
            if "command" in kwargs:
                raise ValueError("Cannot specify both calculator_exec and command")
            if " PREFIX " in calculator_exec:
                raise ValueError("calculator_exec should not include mopac command line arguments such as ' PREFIX.mop > PREFIX.out'")
            self.command = f'{calculator_exec} PREFIX.mop > ' \
                           f'PREFIX.out'
        elif calculator_exec is None and "command" not in kwargs:
            self.command = os.environ.pop("ASE_MOPAC_COMMAND", "mopac PREFIX.mop > PREFIX.out")
        else:
            self.command = kwargs["command"]


        FileIOCalculator.__init__(self, restart, ignore_bad_restart_file,
                                    label, atoms, **kwargs)     

        WFLFileIOCalculator.__init__(self, keep_files=keep_files, rundir_prefix=rundir_prefix,
                                        workdir=workdir, scratchdir=scratchdir,
                                        calculator_exec=calculator_exec)    

        self.extra_results = dict()
        # to make use of wfl.calculators.utils.save_results()
        self.extra_results["atoms"] = {}
        self.extra_results["config"] = {}


    def write_input(self, atoms, properties=None, system_changes=None):

        if self.parameters.get("mult", None) is None:
            self.set(mult=self.get_default_multiplicity(atoms,
                                        self.parameters.get("charge", 0)))

        FileIOCalculator.write_input(self, atoms, properties, system_changes)
        self.parameters.write(self.label + ".ase")

        task = self.task

        if "GRADIENTS" not in task:
            task += " GRADIENTS"

        if "RELSCF" not in task:
            task += " RELSCF=0.0001"

        if self.charge !=0:
            assert isinstance(self.charge, int), "charge must be an integer"
            task += f" CHARGE={int(self.charge)}"

        if self.mult != 1:
            task += f" {self.mult_keywords[self.mult]}"
            if "UHF" not in task:
                task += " UHF"

        with open(self.label + ".mop", "w") as f:
            f.write(task + "\n")
            f.write("Title: ASE calculation\n\n")
            
            for atom in atoms:
                f.write(f" {atom.symbol:2} {atom.position[0]} 1 {atom.position[1]} 1 {atom.position[2]} 1\n")

            if atoms.pbc:
                raise NotImplementedError("pbc not implemented yet")

            
    def read_results(self):
        
        self.check_version()
        self.check_calculation_success():

        self.read_energy()
        self.read_forces()

        if "EF" in self.task:
            self.read_opt_atoms()

    def read_opt_atoms(self):

        with open(self.label + ".out") as fd:
            text = fd.read()  


    def read_forces(self):

        with open(self.label + ".out") as fd:
            text = fd.read() 

        pat_forces_block = re.compile(
            r"FINAL  POINT  AND  DERIVATIVES\s+"
            r"\s+PARAMETER\s+ATOM\s+TYPE\s+VALUE\s+GRADIENT\s+"
            r"((\s+\d+\s+\d+\s+[A-Za-z]+\s+CARTESIAN\s+[XYZ]\s+-?[\.\d]+\s+-?[\.\d]+\s+KCAL/ANGSTROM\n)+)")
        
        pat_force_comp = re.compile(
            r"\s+\d+\s+\d+\s+[A-Za-z]+\s+CARTESIAN\s+[XYZ]\s+-?[\.\d]+\s+(-?[\.\d]+)\s+KCAL/ANGSTROM")
        
        grad_text_lines = pat_forces_block.findall(text)
        assert len(grad_text_lines) == 1, "Could not find forces block in MOPAC output"

        forces = []

        grad_text_lines = grad_text_lines[0][0].split('\n')
        for line in grad_text_lines:
            match = pat_force_comp.search(line)
            forces.append(-float(match.groups()[0]))

        forces = np.array(forces).reshape(-1, 3) * kcal / mol
        self.results["forces"] = forces


    def read_energy(self):

        with open(self.label + ".out") as fd:
            text = fd.read()

        # Developers recommend using final HOF for "energy" as it includes dispersion etc.
        final_heat_regex = re.compile(
            r'^\s+FINAL HEAT OF FORMATION\s+\=\s+(-?\d+\.\d+) KCAL/MOL')

        final_hof = final_heat_regex.search(text)
        self.results["energy"] = float(final_hof.groups()[0]) * kcal / mol


    def check_version(self):

        with open(self.label + ".out") as fd:
            text = fd.read()

        re_version = re.compile(r'\*\*\s+MOPAC (v[\.\d]+)\s+\*\*')
        version =  re_version.search(text)
        if version is not None:
            version = version.groups()[0]
        if version is None or  version.parse(aa) <= version.parse("22"):
            raise ValueError("MOPAC version 22 or greater required")


    def check_calculation_success(self):

        with open(self.label + ".out") as fd:
            text = fd.read()

        re_error = re.compile(r"error", flags=re.IGNORECASE)
        found_lines = re_error.findall(text)
        if len(found_lines) > 0:
            errors = "\n".join(found_lines)
            raise CalculationFailed("MOPAC calculation Errored. Error lines from output: \n" + errors)


    def calculate(self, atoms=None, properties=_default_properties, system_changes=all_changes):
        """Do the calculation. Handles the working directories in addition to regular
        ASE calculation operations (writing input, executing, reading_results)
        Reimplements & extends GenericFileIOCalculator.calculate() for the development version of ASE
        or FileIOCalculator.calculate() for the v3.22.1"""

        # from WFLFileIOCalculator
        self.setup_rundir()

        try:
            super().calculate(atoms=atoms, properties=properties, system_changes=system_changes)
            calculation_succeeded=True
            if 'FAILED_MOPAC' in atoms.info:
                del atoms.info['FAILED_MOPAC']
        except Exception as exc:
            atoms.info['FAILED_MOPAC'] = True
            calculation_succeeded=False
            raise exc
        finally:
            # from WFLFileIOCalculator
            self.clean_rundir(_default_keep_files, calculation_succeeded)


    @staticmethod
    def get_default_multiplicity(atoms, charge=0):
        """ Gets the multiplicity of the atoms object.

        format: singlet = 1, doublet = 2, etc.

        Parameters
        ----------
        atoms : ase.Atoms
        charge : int

        Returns
        -------
        multiplicity : int
        """
        if charge is None:
            charge = 0

        return np.sum(atoms.get_atomic_numbers() - int(charge)) % 2 + 1

