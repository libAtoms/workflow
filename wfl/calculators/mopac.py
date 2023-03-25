"""
MOPAC interface
"""
import re
import os
import numpy as np
from packaging import version

from ase.units import kcal,  mol
from ase.calculators.calculator import CalculationFailed, Calculator, \
    FileIOCalculator, all_changes, InputError

from .wfl_fileio_calculator import WFLFileIOCalculator

_default_keep_files = ["*.out"]
_default_properties = ["energy", "forces"] 

class MOPAC(WFLFileIOCalculator, FileIOCalculator):
    """ Extension of ASE's MOPAC claculator so that it can be used by wfl.calculators.generic (mainly each calculation is run in a separate directory). 
    Currently must request eigenvalue-following geometry optimisation (EF), because 
    single point calculation messes with input geometry somehow. 
    
    Parameters
    ----------
    task: str, default "AM1 EF GRADIENTS RELSCF=0.001
        the first line of input file
    charge: int, default 0
        charge for the calculation
    mult: int, default None
        multiplicity. If none, singlet/doublet gets picked.     
    
    """

    implemented_properties = ["energy", "forces"]

    default_parameters = dict(
        task = "AM1 EF GRADIENTS RELSCF=0.0001",
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
                 calculator_exec=None, 
                 label="mopac", **kwargs):

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

        WFLFileIOCalculator.__init__(self, keep_files=keep_files, rundir_prefix=rundir_prefix,
                                        workdir=workdir, scratchdir=scratchdir,
                                        calculator_exec=calculator_exec, label=label,  **kwargs)    

        self.extra_results = dict()
        # to make use of wfl.calculators.utils.save_results()
        self.extra_results["atoms"] = {}
        self.extra_results["config"] = {}


    def write_input(self, atoms, properties=None, system_changes=None):

        if self.parameters.get("mult", None) is None:
            self.set(mult=self.get_default_multiplicity(atoms,
                                        self.parameters.get("charge", 0)))

        pp = self.parameters

        FileIOCalculator.write_input(self, atoms, properties, system_changes)
        pp.write(self.label + ".ase")

        task = pp.task
        
        # geometry gets changed even when no geometry optimization is requested. 
        # have emailed for help. 
        if "EF" not in task:
            raise InputError("For now MOPAC only supports geometry optimization")

        if "GRADIENTS" not in task:
            task += " GRADIENTS"

        if "RELSCF" not in task:
            task += " RELSCF=0.0001"

        if pp.charge !=0:
            assert isinstance(pp.charge, int), "charge must be an integer"
            task += f" CHARGE={int(pp.charge)}"

        if pp.mult != 1:
            task += f" {self.mult_keywords[pp.mult]}"
            if "UHF" not in task:
                task += " UHF"

        with open(self.label + ".mop", "w") as f:
            f.write(task + "\n")
            f.write("Title: ASE calculation\n\n")
            
            for atom in atoms:
                f.write(f" {atom.symbol:2} {atom.position[0]} 1 {atom.position[1]} 1 {atom.position[2]} 1\n")

            if np.any(atoms.pbc):
                raise NotImplementedError("pbc not implemented yet")

            
    def read_results(self):
        
        self.check_version()
        self.check_calculation_success()

        self.read_energy()
        self.read_forces()

        if "EF" in self.parameters.task:
            self.read_opt_atoms()

    def read_opt_atoms(self):

        with open(self.label + ".out") as fd:
            text = fd.read()  

        pat_geometries_block = re.compile(
            r"CARTESIAN COORDINATES\n\n"
             r"((\s+\d+\s+[A-Za-z]+\s+[-\.\d]+\s+[-\.\d]+\s+[-\.\d]+\n)+)")

        pat_coord = re.compile(r"\s+\d+\s+[A-Za-z]+\s+([-\.\d]+)\s+([-\.\d]+)\s+([-\.\d]+)")
        opt_positions = []

        geometry_lines = pat_geometries_block.findall(text)
        geometry_lines = geometry_lines[0][0].split('\n')
        for line in geometry_lines: 
            if len(line) == 0:
                continue
            coord_match = pat_coord.search(line)
            opt_positions += [float(num) for num in coord_match.groups()]

        opt_positions = np.array(opt_positions).reshape(-1, 3)
        self.extra_results["relaxed_positions"] = opt_positions


    def read_forces(self):

        with open(self.label + ".out") as fd:
            text = fd.read() 

        pat_forces_block = re.compile(
            r"FINAL  POINT  AND  DERIVATIVES\n\n"
            r"\s+PARAMETER\s+ATOM\s+TYPE\s+VALUE\s+GRADIENT\n"
            r"((\s+\d+\s+\d+\s+[A-Za-z]+\s+CARTESIAN\s+[XYZ]\s+-?[\.\d]+\s+-?[\.\d]+\s+KCAL/ANGSTROM\n)+)")
        
        pat_force_comp = re.compile(
            r"\s+\d+\s+\d+\s+[A-Za-z]+\s+CARTESIAN\s+[XYZ]\s+-?[\.\d]+\s+(-?[\.\d]+)\s+KCAL/ANGSTROM")
        
        grad_text_lines = pat_forces_block.search(text)

        forces = []
        grad_text_lines = grad_text_lines.groups()[0].split('\n')
        for line in grad_text_lines:
            if len(line) == 0:
                continue
            match = pat_force_comp.search(line)
            forces.append(-float(match.groups()[0]))

        forces = np.array(forces).reshape(-1, 3) * kcal / mol
        self.results["forces"] = forces


    def read_energy(self):

        with open(self.label + ".out") as fd:
            text = fd.read()

        # Developers recommend using final HOF for "energy" as it includes dispersion etc.
        final_heat_regex = re.compile(
            r"FINAL HEAT OF FORMATION =\s+([-\.\d]+) KCAL/MOL =\s+[-\.\d]+ KJ/MOL")

        final_hof = final_heat_regex.findall(text)
        assert len(final_hof) == 1
        self.results["energy"] = float(final_hof[0]) * kcal / mol


    def check_version(self):

        with open(self.label + ".out") as fd:
            text = fd.read()

        re_version = re.compile(r'\*\*\s+MOPAC (v[\.\d]+)\s+\*\*')
        mopac_version =  re_version.search(text)
        if mopac_version is not None:
            mopac_version = mopac_version.groups()[0]
        if mopac_version is None or  version.parse(mopac_version) <= version.parse("22"):
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

