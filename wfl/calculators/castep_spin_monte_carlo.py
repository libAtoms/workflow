"""
Implementation of "Spin-Committee" global optimisation scheme
for CASTEP spin polarised calculations.

Original code and idea by Vlad Carare, then generalised
by T. K. Stenczel here.

Scheme:
- initialise n_samples number of calculations with
randomly chosen initial spin
- take the minimum energy configuration for the results
"""
from typing import List

import ase
import numpy as np
from ase.calculators.calculator import Calculator, all_changes
from ase.units import kB

from .castep import evaluate_op


# TODO: convert the initializer to an ABC and subclass for possible different initialisation methods
# todo: look up is VASP & QE can use the magnetic initialisation, because then we can have a general calculator


class UniformSpinInitializer:
    """Initialises spin from a uniform random distribution"""

    def __init__(self, low: float, high: float):
        self.low = low
        self.high = high

    def sample_multiple(
        self, atoms: ase.Atoms, number: int
    ) -> List[ase.Atoms]:
        assert number > 0
        return [self.sample(atoms) for _ in range(number)]

    def sample(self, atoms: ase.Atoms) -> ase.Atoms:
        new_atoms = atoms.copy()

        # random uniform spins
        spins = np.random.uniform(self.low, self.high, size=len(atoms))

        new_atoms.set_initial_magnetic_moments(spins)

        return new_atoms


class CastepSpinMonteCarlo(Calculator):
    """
    nb. this needs https://gitlab.com/ase/ase/-/merge_requests/2464
    to be finished in order to work
    """

    implemented_properties = ["energy", "forces", "stress"]

    def __init__(
        self,
        atoms=None,
        n_samples: int = 15,
        spin_min: float = -1.0,
        spin_max: float = 1.0,
        method: str = "min",
        boltzmann_t: float = 0.0,
        # now the Castep keywords
        base_rundir=None,
        dir_prefix="run_CASTEP_",
        calculator_command=None,
        calculator_kwargs=None,
        output_prefix="CASTEP_",
        properties=None,
        keep_files="default",
        **kwargs,
    ):
        super().__init__(atoms=atoms, **kwargs)
        self.n_samples = n_samples
        self.spin_initializer = UniformSpinInitializer(
            low=spin_min, high=spin_max
        )
        self.method = method
        self.boltzmann_t = boltzmann_t

        # this will be passed to the operation class
        self._castep_op_kw = dict(
            base_rundir=base_rundir,
            dir_prefix=dir_prefix,
            calculator_command=calculator_command,
            calculator_kwargs=calculator_kwargs,
            output_prefix=output_prefix,
            properties=properties,
            keep_files=keep_files,
        )

    @property
    def method(self):
        return self._method

    @method.setter
    def method(self, value: str):
        if value.lower() not in ["mean", "min", "boltzmann"]:
            raise ValueError(f"Method not understood: {value}")
        self._method = value.lower()

    def calculate(
        self, atoms=None, properties=None, system_changes=all_changes
    ):
        if properties is None:
            properties = self.implemented_properties

        # needed dry run of the ase calculator and check if calculation is required
        ase.calculators.calculator.Calculator.calculate(
            self, atoms, properties, system_changes
        )
        if not self.calculation_required(self.atoms, properties):
            return

        # create N copies of the atoms object
        guesses = self.spin_initializer.sample_multiple(
            self.atoms, self.n_samples
        )

        # do a calculation on all of them
        results = evaluate_op(guesses, **self._castep_op_kw)
        return self.choose_result(results, properties)

    def choose_result(
        self, atoms_list: List[ase.Atoms], properties=None
    ) -> ase.Atoms:
        if properties is None:
            properties = self.implemented_properties

        if self.method == "min":
            # choose the lowest energy one
            def _key(x):
                # get energy, if it does not exist then return inf
                # so we don't choose this one
                parameter_name = (
                    self._castep_op_kw.get("output_prefix") + "energy"
                )
                return x.info.get(parameter_name, np.inf)

            minimum_energy_atoms = min(atoms_list, key=_key)
            return minimum_energy_atoms
        elif self.method == "mean":
            return self._linear_combination_of_results(
                atoms_list, properties, None
            )
        elif self.method == "boltzmann":
            key = self._castep_op_kw.get("output_prefix") + "energy"

            # calculate Boltzmann weights
            energies = np.array([at.info.get(key) for at in atoms_list])
            e_kt = np.exp(
                -(energies - np.min(energies)) / (kB * self.boltzmann_t)
            )
            weights = e_kt / np.sum(e_kt)

            return self._linear_combination_of_results(
                atoms_list, properties, weights
            )

        else:
            raise ValueError(
                "Method not understood, have you "
                "touched a protected member perhaps"
            )

    def _linear_combination_of_results(
        self, atoms_list: List[ase.Atoms], properties, weights=None
    ) -> ase.Atoms:

        assert len(atoms_list) > 0

        new_atoms = atoms_list[0].copy()

        if "energy" in properties:
            # energy
            key = self._castep_op_kw.get("output_prefix") + "energy"
            new_atoms.info[key] = np.average(
                [at.info.get(key) for at in atoms_list], weights=weights
            )

        if "forces" in properties:
            # forces
            key = self._castep_op_kw.get("output_prefix") + "forces"
            new_atoms.arrays[key] = np.average(
                [at.arrays.get(key) for at in atoms_list],
                axis=0,
                weights=weights,
            )

        if "stress" in properties:
            # stress -> (6, ) vector
            key = self._castep_op_kw.get("output_prefix") + "stress"
            new_atoms.info[key] = np.average(
                [at.info.get(key) for at in atoms_list],
                axis=0,
                weights=weights,
            )

        return new_atoms
