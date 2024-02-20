"""
Generate configs with normal mode sampling, generally useful for small
molecules to learn around their relaxed state.
"""

import os
import warnings
from copy import deepcopy

import ase.io
import math
import numpy as np
from ase import Atoms, units
from scipy import stats

from wfl.calculators import generic
from wfl.configset import ConfigSet, OutputSpec
from wfl.autoparallelize import autoparallelize, autoparallelize_docstring
from wfl.autoparallelize import AutoparaInfo
from wfl.utils.misc import atoms_to_list

# conversion factor from eV/Å^2/amu to eV^2
eigenval_units_factor = units._hbar ** 2 * 1e10 ** 2 / (units._e *
                                                        units._amu)

class NormalModes:
    # displacement for generating numerical hessian
    num_hess_delta = 0.01

    def __init__(self, nm_atoms, prop_prefix):
        """ Allows to do normal mode-related operations.

        Parameters
        ----------

        nm_atoms: str / Atoms
            Atoms or xyz file with single Atoms with normal mode information
        prop_prefix: str / None
            prefix for normal_mode_frequencies and normal_mode_displacements
            stored in atoms.info/arrays

        Returns
        -------

        """

        self.prop_prefix = prop_prefix

        if isinstance(nm_atoms, str):
            self.atoms = ase.io.read(nm_atoms)
        else:
            self.atoms = nm_atoms

        self.num_at = len(self.atoms)
        self.num_nm = self.num_at * 3
        self.inverse_m = np.repeat(self.atoms.get_masses() ** -0.5, 3)

        # dynamical matrix's eigenvalues in eV/Å^2/amu
        self.eigenvalues = None

        # dynamical matrix's eigenvectors in Å * sqrt(amu)
        # normalised and orthogonal
        self.eigenvectors = np.zeros((3 * self.num_at, 3 * self.num_at))

        # normal mode displacements in Å
        self.modes = np.zeros((self.num_nm, self.num_at, 3))

        # normal mode frequencies in eV - square root of eigenvalue
        self.frequencies = np.zeros(self.num_nm)

        if f'{self.prop_prefix}normal_mode_frequencies' in \
                self.atoms.info.keys():
            self._collect_nm_info_from_atoms()

    def _collect_nm_info_from_atoms(self):
        """Collects normal mode frequencies and displacements from the Atoms
         object and converts them to eigenvalues/vectors needed for random
         displacements."""

        self.frequencies = self.atoms.info[
            f'{self.prop_prefix}normal_mode_frequencies']

        for idx in range(self.num_nm):
            self.modes[idx] = self.atoms.arrays[
                f'{self.prop_prefix}normal_mode_displacements_{idx}']

        self.eigenvalues = self.freqs_to_evals(self.frequencies)
        self.eigenvectors = self.modes_to_evecs(self.modes,
                                                inverse_m=self.inverse_m)


    @staticmethod
    def freqs_to_evals(freqs):
        """Converts from frequencies (sqrt of eigenvalue) in eV to eigenvalues
        in eV/Å^2/amu. Negative frequencies are shorthand for imaginary
        frequencies. """

        evals = []
        for freq in freqs:

            if freq < 0:
                factor = -1
            else:
                factor = 1

            evals.append(factor * freq ** 2 / eigenval_units_factor)

        return np.array(evals)

    @staticmethod
    def evals_to_freqs(evals):
        """Converts from eigenvalues in eV/Å^2/amu to frequencies (square
         root of eigenvalue) in eV. Negative frequencies are shorthand for
         imaginary frequencies."""

        frequencies = []
        for eigenvalue in evals:

            freq = (eigenval_units_factor *
                    eigenvalue.astype(complex)) ** 0.5

            if np.imag(freq) != 0:
                freq = -1 * np.imag(freq)
            else:
                freq = np.real(freq)

            frequencies.append(freq)

        return np.array(frequencies)

    @staticmethod
    def evecs_to_modes(evecs, masses=None, inverse_m=None):
        """converts from mass-weighted 3N-long eigenvector to 3N displacements
         in Cartesian coordinates"""

        assert masses is None or inverse_m is None

        if masses is not None:
            inverse_m = np.repeat(masses ** -0.5, 3)

        n_free = len(evecs)
        modes = evecs * inverse_m
        # normalise before reshaping.
        # no way to select axis when dividing, so have to transpose,
        # normalise, transpose.
        norm = np.linalg.norm(modes.T, axis=0)
        modes = np.divide(modes.T, norm).T
        modes = modes.reshape(n_free, int(n_free / 3), 3)
        return modes

    @staticmethod
    def modes_to_evecs(modes, masses=None, inverse_m=None):
        """converts 3xN cartesian displacements to 1x3N mass-weighted
        eigenvectors"""

        assert masses is None or inverse_m is None

        if masses is not None:
            inverse_m = np.repeat(masses ** -0.5, 3)

        n_free = len(inverse_m)
        eigenvectors = modes.reshape(n_free, n_free)
        eigenvectors /= inverse_m

        # normalise
        # no way to select axis when dividing,
        # so have to transpose, normalise, transpose.
        norm = np.linalg.norm(eigenvectors, axis=1)
        eigenvectors = np.divide(eigenvectors.T, norm).T

        return eigenvectors

    def summary(self):
        """Prints all vibrational frequencies.  """

        print('---------------------\n')
        print('  #    meV     cm^-1\n')
        print('---------------------\n')
        for idx, en in enumerate(self.frequencies):
            if en < 0:
                c = ' i'
                en = np.abs(en)
            else:
                c = '  '

            print(f'{idx:3d} {1000 * en:6.1f}{c} {en / units.invcm:7.1f}{c}')
        print('---------------------\n')

    def view(self, prefix='nm', output_dir='normal_modes',
                   normal_mode_numbers='all', temp=300, nimages=16):
        """writes out xyz files with oscillations along each of the normal
        modes

        Parameters
        ----------

        prefix: str, default "nm"
            Prefix normal mode files
        output_dir: str, default "normal_modes"
            Directory for outputs
        normal_mode_numbers: str / list(int) / np.array(int), default "all"
            List of normal mode numbers to write out or "all" to save all
        temp: float, default 300
            Temperature for the oscillation
        nimages: int, default 32
            Number of structures per oscillation/output file

        Returns
        -------
        """

        if normal_mode_numbers == 'all':
            normal_mode_numbers = np.arange(self.num_nm)
        elif isinstance(normal_mode_numbers, int):
            normal_mode_numbers = [normal_mode_numbers]
        if not os.path.isdir(output_dir):
            os.makedirs(output_dir)

        for nm in normal_mode_numbers:
            mode = self.modes[nm] * math.sqrt(
                units.kB * temp / abs(self.frequencies[nm]))
            traj = os.path.join(output_dir, f'{prefix}_{nm}.xyz')
            if os.path.isfile(traj):
                os.remove(traj)
            for x in np.linspace(0, 2 * math.pi, nimages, endpoint=False):
                at = self.atoms.copy()
                at.positions += math.sin(x) * mode.reshape((self.num_at, 3))
                at.write(traj, append=True)

    def sample_normal_modes(self, sample_size, temp=None,
                            energies_for_modes=None,
                            normal_mode_numbers='all', info_to_keep="default",
                            arrays_to_keep=None):
        """ Randomly displace equilibrium structure's atoms along given
        normal modes so that normal mode energies follow Boltzmann
        distribution at a given temperature.

        Notes
        -----

        Coefficients for scaling the individual mass-weighted normal
        coordinates are sampled from a normal distribution with standard
        deviation of  eigenvalue / (kB * temp). The harmonic energy
        distribution then follows a generalised gamma distribution with
        degrees of freedom = len(normal_mode_numbers) or
        shape = len(normal_mode_numbers) / 2.

        Returned atoms have an atoms.info['nm_energy'] entry with the
        corresponding harmonic energy


        Parameters
        ----------

        sample_size: int
            How many randomly perturbed structures to return

        temp: float, default None
            Temperature for the displacement (putting kT energy into each
            mode) alternative to `energies_for_modes`

        energies_for_modes: list(float), default: None
            list of energies (e.g. kT) for each of the normal modes
            for generating displacement magnitudes. Alternative to `temp`.
            Length either 3N - 6 if normal_mode_numbers == 'all' or matches
            len(normal_mode_numbers).

        normal_mode_numbers:  int / list(int) / 'all', default "all"
            List of normal mode numbers to displace along. Alternatively if
            "all" is selected, all but first six (rotations and translations)
            normal modes are used.

        info_to_keep: list(str) / None / 'default', default 'default'
            list of Atoms.info.keys() to keep, defaults to 'config_type' only

        arrays_to_keep: list(str), default None
            list of Atoms.arrays.keys() entries to keep


        Returns
        -------
        list(Atoms) of randomly perturbed structures
        """

        assert temp is None or energies_for_modes is None

        if isinstance(normal_mode_numbers, str):
            if normal_mode_numbers == 'all':
                normal_mode_numbers = np.arange(6, self.num_at * 3)
        elif isinstance(normal_mode_numbers, int):
            normal_mode_numbers = [normal_mode_numbers]

        if info_to_keep == 'default':
            info_to_keep = ['config_type']
        elif info_to_keep is None:
            info_to_keep = []

        if arrays_to_keep is None:
            arrays_to_keep = []

        if energies_for_modes is not None:
            assert len(energies_for_modes) == len(normal_mode_numbers)
            if isinstance(energies_for_modes, list):
                energies_for_modes = np.array(energies_for_modes)

        elif temp is not None:
            energies_for_modes = np.array([units.kB * temp] *
                                          len(normal_mode_numbers))

        n = len(normal_mode_numbers)

        cov = np.eye(n) * energies_for_modes / self.eigenvalues[normal_mode_numbers]
        norm = stats.multivariate_normal(mean=np.zeros(n), cov=cov,
                                         allow_singular=True)

        alphas_list = norm.rvs(size=sample_size)
        if sample_size == 1:
            alphas_list = [alphas_list]

        sampled_configs = []
        for alphas in alphas_list:
            if len(normal_mode_numbers) == 1:
                alphas = [alphas]

            individual_displacements = np.array([aa * evec for aa, evec
                     in zip(alphas, self.eigenvectors[normal_mode_numbers])])

            mass_wt_displs = individual_displacements.sum(axis=0)
            displacements = mass_wt_displs * self.inverse_m
            displacements = displacements.reshape(len(self.atoms), 3)

            new_pos = self.atoms.positions.copy() + displacements
            symbols = self.atoms.symbols

            displaced_at = Atoms(symbols, positions=new_pos)

            for info_key in info_to_keep:
                try:
                    displaced_at.info[info_key] = deepcopy(
                        self.atoms.info[info_key])
                except KeyError:
                    continue
            for arrays_key in arrays_to_keep:
                try:
                    displaced_at.arrays[arrays_key] = deepcopy(
                        self.atoms.arrays[arrays_key])
                except KeyError:
                    continue

            energy = sum([aa ** 2 * eigenval / 2 for aa, eigenval in
                          zip(alphas, self.eigenvalues[normal_mode_numbers])])

            displaced_at.info[f'{self.prop_prefix}normal_mode_energy'] = energy

            if temp is not None:
                displaced_at.info[f'{self.prop_prefix}normal_mode_temperature'] = temp

            sampled_configs.append(displaced_at)
        return sampled_configs

    def derive_normal_mode_info(self, calculator, parallel_hessian=True):
        """Get normal mode information using numerical hessian

        Parameters
        ----------

        calculator: Calculator / (initializer, args, kwargs)
            ASE calculator or routine to call to create calculator
        parallel_hessian: bool, default=True
            whether to parallelize 6N calculations needed for approximating
            the Hessian.

        Returns
        -------
        """

        displaced_in_configset = ConfigSet(self._displace_at_in_xyz())
        displaced_out_configset = OutputSpec()

        properties = ['energy', 'forces']

        if parallel_hessian:
            generic.calculate(
                inputs=displaced_in_configset,
                outputs=displaced_out_configset,
                calculator=calculator,
                output_prefix=self.prop_prefix,
                properties=properties,
                autopara_info=AutoparaInfo(num_inputs_per_python_subprocess=1))

            self._write_nm_to_atoms(
                displaced_ats=list(displaced_out_configset.to_ConfigSet()))

        else:
            displaced_out_atoms = generic._run_autopara_wrappable(atoms=displaced_in_configset,
                                             calculator=calculator,
                                             properties=properties,
                                             output_prefix=self.prop_prefix)
            self._write_nm_to_atoms(displaced_ats=displaced_out_atoms)


    def _write_nm_to_atoms(self, displaced_ats):
        """ Assigns reference atoms their normal mode information

        Parameters
        ---------
        displaced_ats: list(Atoms)
            Atoms displaced along each Cartesian coordinate for the
                approximation

        Returns
        -------

        """

        hessian = np.empty((self.num_nm, self.num_nm))

        for idx, (at_minus, at_plus) in enumerate(
                zip(displaced_ats[0::2], displaced_ats[1::2])):

            p_name = at_plus.info['disp_direction']
            m_name = at_minus.info['disp_direction']

            if ('+' not in p_name) or ('-' not in m_name) or (
                    p_name[:-1] != m_name[:-1]):
                raise RuntimeError(
                    f'The displacements are not what I think they are, '
                    f'got: {p_name} and {m_name}')

            f_plus = at_plus.arrays[f'{self.prop_prefix}forces']
            f_minus = at_minus.arrays[f'{self.prop_prefix}forces']

            hessian[idx] = (f_minus - f_plus).ravel() / 4 / \
                           self.num_hess_delta

        hessian += hessian.copy().T
        e_vals, e_vecs = np.linalg.eigh(
            np.array([self.inverse_m]).T * hessian * self.inverse_m)

        self.eigenvalues = e_vals
        self.eigenvectors = e_vecs.T

        self.frequencies = self.evals_to_freqs(self.eigenvalues)
        self.modes = self.evecs_to_modes(self.eigenvectors,
                                                inverse_m=self.inverse_m)

        self._update_atoms_info_and_arrays()

    def _update_atoms_info_and_arrays(self):
        """Assigns normal modes and frequencies to atoms.info/arrays."""

        freq_key = f'{self.prop_prefix}normal_mode_frequencies'
        if freq_key in self.atoms.info.keys():
            warnings.warn(f'overwriting {freq_key} & associated modes present'
                          f'in self.atoms')

        for freq in self.frequencies[6:]:
            freq *= units.invcm
            if freq < 0:
                warnings.warn(f'Found imaginary frequency of {freq} cm^-1')

        self.atoms.info[freq_key] = self.frequencies
        for idx in range(self.num_nm):
            self.atoms.arrays[
                f'{self.prop_prefix}normal_mode_displacements_{idx}'] = \
                self.modes[idx]

    def _displace_at_in_xyz(self):
        """ displace each atom along each of xyz backwards and forwards
        for approximating the Hessian.
        """
        displaced_ats = []

        for disp_name, index, coordinate, displacement in \
                self._yield_displacements():
            at_copy = self.atoms.copy()
            at_copy.positions[index, coordinate] += displacement
            at_copy.info['disp_direction'] = disp_name
            displaced_ats.append(at_copy)

        return displaced_ats

    def _yield_displacements(self):
        """modified ase.NormalModes.NormalModes"""

        indices = np.arange(self.num_at)
        for index in indices:
            for coord_idx, coord in enumerate('xyz'):
                for sign in [-1, 1]:
                    if sign == -1:
                        sign_symbol = '-'
                    else:
                        sign_symbol = '+'
                    disp_name = f'{index}{coord}{sign_symbol}'
                    displacement = sign * self.num_hess_delta
                    yield disp_name, index, coord_idx, displacement


def sample_normal_modes(inputs, outputs, temp, sample_size, prop_prefix,
                        info_to_keep=None, arrays_to_keep=None):
    """Multiple times displace along normal modes for all atoms in input

    Parameters
    ----------

    inputs: Atoms / list(Atoms) / ConfigSet
        Structures with normal mode information (eigenvalues & eigenvectors)
    outputs: OutputSpec
    temp: float
        Temperature for normal mode displacements
    sample_size: int
        Now many perturbed structures per input structure to return
    prop_prefix: str / None
        prefix for normal_mode_frequencies and normal_mode_displacements
        stored in atoms.info/arrays
    info_to_keep: str, default "config_type"
        string of Atoms.info.keys() to keep
    arrays_to_keep: str, default None
        string of Atoms.arrays.keys() entries to keep

    Returns
    -------
      """

    if isinstance(inputs, Atoms):
        inputs = [inputs]

    for atoms in inputs:
        at_vib = NormalModes(atoms, prop_prefix)
        sample = at_vib.sample_normal_modes(temp=temp,
                                            sample_size=sample_size,
                                            info_to_keep=info_to_keep,
                                            arrays_to_keep=arrays_to_keep)
        outputs.store(sample)

    outputs.close()


def _generate_normal_modes_autopara_wrappable(inputs, calculator, prop_prefix,
                             parallel_hessian):
    """Get normal mode information for all atoms in the input

    Parameters
    ----------
    inputs: Atoms / list(Atoms)
        input configs
    calculator: Calculator / (initializer, args, kwargs)
        ASE calculator or routine to call to create calculator
    prop_prefix: str / None
        prefix for normal_mode_frequencies and normal_mode_displacements
        stored in atoms.info/arrays
    parallel_hessian: bool, default=True
        whether to parallelize 6N calculations needed for Hessian
        approximation.

    Returns
    -------
    list(Atoms) with normal mode information
    """

    atoms_out = []
    for atoms in atoms_to_list(inputs):
        at_vib = NormalModes(atoms, prop_prefix)
        at_vib.derive_normal_mode_info(calculator=calculator,
                                       parallel_hessian=parallel_hessian)
        atoms = at_vib.atoms
        atoms_out.append(atoms)

    return atoms_out


def generate_normal_modes_parallel_atoms(*args, **kwargs):
    # iterable loop parallelizes over input structures, not over 6xN
    # displaced structures needed for numerical hessian
    kwargs["parallel_hessian"] = False
    return autoparallelize(_generate_normal_modes_autopara_wrappable, *args,
                           default_autopara_info={"num_inputs_per_python_subprocess": 10}, **kwargs)
autoparallelize_docstring(generate_normal_modes_parallel_atoms, _generate_normal_modes_autopara_wrappable, "Atoms")


def generate_normal_modes_parallel_hessian(inputs, outputs, calculator,
                                           prop_prefix):
    parallel_hessian = True
    atoms_out = _generate_normal_modes_autopara_wrappable(inputs=inputs,
                                         calculator=calculator,
                                         prop_prefix=prop_prefix,
                                         parallel_hessian=parallel_hessian)

    outputs.store(atoms_out)
    outputs.close()
