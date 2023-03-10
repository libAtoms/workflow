"""
Committee of Models

Calculated properties with a list of models and saves them into info/arrays.
For `calculate_committee() further operations (eg. mean, variance, etc.) with these are up to the user.
"""
import warnings
import numpy as np
from collections import Counter

from ase import Atoms
import ase.io
from ase.calculators.calculator import Calculator, all_changes
from ase.calculators.calculator import PropertyNotImplementedError

from .utils import per_atom_properties, per_config_properties
from ..utils.misc import atoms_to_list
from ..utils.parallel import construct_calculator_picklesafe


__default_properties = ['energy', 'forces', 'stress']


def calculate_committee(atoms, calculator_list, properties=None, output_prefix="committee_{}_"):
    """Calculate energy and forces with a committee of models

    Notes
    -----
    Supports formatter string in the output_prefix arg, but currently only with a single field literally "{}".

    Parameters
    ----------
    atoms : Atoms / list(Atoms)
        input atomic configs
    calculator_list : list(Calculator) / list[(initializer, args, kwargs)]
        list of calculators to use as a committee of models on the configs
    properties: list[str], default ['energy', 'forces', 'stress']
        properties to calculate
    output_prefix : str, default="committee\_"
        prefix for results coming from the committee of models.
        If includes "{}" then will use it as a format string, otherwise puts a number at the end of prefix for the
        index of the model in the committee of models

    Returns
    -------
    atoms : Atoms / list(Atoms)

    """
    if properties is None:
        properties = __default_properties

    atoms_list = atoms_to_list(atoms)

    # create the general key formatter
    if "{}" in output_prefix:
        if output_prefix.count("{}") != 1:
            raise ValueError("Prefix with formatting is incorrect, cannot have more than one of `{}` in it")
        key_formatter = f"{output_prefix}{{}}"
    else:
        key_formatter = f"{output_prefix}{{}}{{}}"

    # create calculator instances
    calculator_list_to_use = [construct_calculator_picklesafe(calc) for calc in calculator_list]

    for at in atoms_list:
        # calculate forces and energy with all models from the committee
        for i_model, pot in enumerate(calculator_list_to_use):
            for prop in properties:
                if prop in per_atom_properties:
                    at.arrays[key_formatter.format(i_model, prop)] = pot.get_property(name=prop, atoms=at)
                elif prop in per_config_properties:
                    at.info[key_formatter.format(i_model, prop)] = pot.get_property(name=prop, atoms=at)
                else:
                    raise ValueError("Don't know where to put property: {}".format(prop))

    if isinstance(atoms, Atoms):
        return atoms_list[0]
    else:
        return atoms_list


class CommitteeUncertainty(Calculator):
    """
    Calculator for a committee of machine learned interatomic potentials (MLIP).

    The class assumes individual members of the committee already exist (i.e. their
    training is performed externally). Instances of this class are initialized with
    these committee members and results (energy, forces) are calculated as average
    over these members. In addition to these values, also the uncertainty (standard
    deviation) is caculated.

    The idea for this Calculator class is based on the following publication:
    Musil et al., J. Chem. Theory Comput. 15, 906−915 (2019)
    https://pubs.acs.org/doi/full/10.1021/acs.jctc.8b00959
    """

    def __init__(self, committee_calculators, atoms=None):
        """Implementation of sum of calculators.

        committee_calculators: list(N)
            Collection of Calculators representing the committee.
        committee_filenames: list(N)
            Collection of paths to subsampled sets for training committee Calculators.
        atoms : ase-Atoms
            Optional object to which the calculator will be attached.
        """
        self.__name__ = 'CommitteeUncertainty'
        self.implemented_properties = ['energy', 'forces', 'stress']
        self._alphas = dict()

        self.committee = Committee(
                members=[CommitteeMember(c_i) for c_i in committee_calculators]
                )

        super().__init__(atoms=atoms)

    def calibrate(self, properties, keys_ref, committee_filenames, appearance_threshold, system_changes=all_changes):
        self.committee.add_training_data(committee_filenames)
        self.committee.set_internal_validation_set(appearance_threshold)
        self.committee.calibrate(properties, keys_ref, system_changes=all_changes)

    def calculate(self, atoms=None, properties=['energy', 'forces', 'stress'], system_changes=all_changes):
        """Calculates committee (mean) values and variances."""

        super().calculate(atoms, properties, system_changes)

        property_committee = {k_i: [] for k_i in properties}

        for cm_i in self.committee.members:
            cm_i.calculator.calculate(atoms=atoms, properties=properties, system_changes=system_changes)

            for p_i in properties:
                property_committee[p_i].append(cm_i.calculator.results[p_i])

        for p_i in properties:
            self.results[p_i] = np.mean(property_committee[p_i], axis=0)
            self.results[f'{p_i}_uncertainty'] = np.sqrt(np.var(property_committee[p_i], ddof=1, axis=0))

            if self.committee.is_calibrated_for(p_i):
                self.results[f'{p_i}_uncertainty'] = self.committee.scale(self.results[f'{p_i}_uncertainty'], p_i)
            else:
                warnings.warn(f'Uncertainty estimation has not been calibrated for {p_i}.')


class Committee:
    """
    Current implementation for linear scaling only.
    """
    def __init__(self, members=[]):
        self.members = members
        self._update()

    @property
    def number(self):
        return self._number

    @property
    def atoms(self):
        return self._atoms

    @property
    def ids(self):
        return self._ids

    @property
    def id_to_atoms(self):
        return self._id_to_atoms

    @property
    def id_counter(self):
        return self._id_counter

    @property
    def alphas(self):
        return self._alphas

    @property
    def calibrated_for(self):
        return self._calibrated_for

    @property
    def validation_set(self):
        if not self._get_new_validation_required:
            return self._validation_set
        else:
            raise PropertyNotImplementedError('`Committee`-instance has been altered since last call of `Committee.set_internal_validation_set()`.')

    def _update(self):
        self._number = len(self.members)
        self._atoms = [atoms_ij for cm_i in self.members for atoms_ij in cm_i.atoms]
        self._ids = [id_ij for cm_i in self.members for id_ij in cm_i.ids]
        self._id_to_atoms = {id_i: atoms_i for id_i, atoms_i in zip(self.ids, self.atoms)}
        self._id_counter = Counter(self.ids)
        self._get_new_validation_required = True
        self._alphas = {}
        self._calibrated_for = set()

    def is_calibrated_for(self, prop):
        return prop in self._calibrated_for

    def __add__(self, member):
        self.members.append(member)
        self._update()

    def __repr__(self):
        s = ''

        s_i = f'Committee Status\n'
        s += s_i
        s += '='*len(s_i) + '\n\n'

        s += f'# members:                    {self.number:>10d}\n'
        s += f'# atoms:                      {len(self.atoms):>10d}\n'
        s += f'# ids:                        {len(self.ids):>10d}\n'
        s += f'# new validation required:    {self._get_new_validation_required!r:>10}\n'
        s += f'# calibrated for:\n'
        for p_i in sorted(self.calibrated_for):
            s += f'{"":>30s}{p_i:>10}\n'

        for idx_i, cm_i in enumerate(self.members):
            s += '\n\n'
            s_i = f'Committee Member {idx_i}:\n'
            s += s_i
            s += '-'*len(s_i) + '\n'
            s += cm_i.__repr__()

        return s

    def add_training_data(self, filenames):
        assert len(filenames) == self.number

        for cm_i, f_i in zip(self.members, filenames):
            cm_i.add_training_data(f_i)
        self._update()

    def set_internal_validation_set(self, appearance_threshold):

        assert appearance_threshold <= self.number - 2

        self._validation_set = []
        for id_i, appearance_i in self.id_counter.most_common()[::-1]:
            if appearance_i > appearance_threshold:
                break
            self._validation_set.append(self.id_to_atoms[id_i])
        self._get_new_validation_required = False

    def calibrate(self, properties, keys_ref, system_changes=all_changes):

        validation_pred = {p_i: np.empty(len(self.validation_set)) for p_i in properties}
        validation_pred_var = {f'{p_i}_variance': np.empty(len(self.validation_set)) for p_i in properties}

        for idx_i, sample_i in enumerate(self.validation_set):

            sample_committee_pred = {p_i: [] for p_i in properties}

            for cm_i in self.members:

                if cm_i.is_sample_in_atoms(sample_i):
                    continue

                cm_i.calculator.calculate(atoms=sample_i, properties=properties, system_changes=system_changes)
                for p_i in properties:
                    sample_committee_pred[p_i].append(cm_i.calculator.results[p_i])

            # assert len(sample_committee_pred[p_i]) > 1, f'Not enough samples '

            for p_i in properties:
                validation_pred[p_i][idx_i] = np.mean(sample_committee_pred[p_i])
                validation_pred_var[f'{p_i}_variance'][idx_i] = np.var(sample_committee_pred[p_i], ddof=1, axis=0)

        validation_ref = {p_i: np.array([atoms_i.info[k_i] for atoms_i in self.validation_set]) for p_i, k_i in zip(properties, keys_ref)}

        for p_i in properties:
            self._alphas.update(
                    {p_i: self._get_alpha(vals_ref=validation_ref[p_i],
                                          vals_pred=validation_pred[p_i],
                                          var_pred=validation_pred_var[f'{p_i}_variance'],
                                          M=self.number
                                          )
                    })
            self._calibrated_for.add(p_i)

    def _get_alpha(self, vals_ref, vals_pred, var_pred, M):
        "Get scaling factor alpha."
        N_val = len(vals_ref)
        alpha_squared = -1/M + (M - 3)/(M - 1) * 1/N_val * np.sum(np.power(vals_ref-vals_pred, 2) / var_pred)
        assert alpha_squared > 0, f'Obtained negative value for `alpha_squared`: {alpha_squared}'
        return np.sqrt(alpha_squared)

    def scale(self, value, prop):
        return self.alphas[prop] * value


class CommitteeMember:
    def __init__(self, calculator, filename=None):  # TODO: Allow both `filename` and list(Atoms)
        self._calculator = calculator
        self._filename = filename

        if self._filename is None:
            self._atoms = []
            self._ids = []
        else:
            self._update()

    @property
    def calculator(self):
        return self._calculator

    @property
    def filename(self):
        return self._filename

    @property
    def atoms(self):
        return self._atoms

    @property
    def ids(self):
        return self._ids

    def _update(self):
        self._atoms = ase.io.read(self.filename, ':')
        self._ids = [atoms_i.info['_ConfigSet_loc__FullTraining'] for atoms_i in self.atoms]

    def add_training_data(self, filename):
        self._filename = filename
        self._update()

    def is_sample_in_atoms(self, sample):
        return sample.info['_ConfigSet_loc__FullTraining'] in self.ids

    def __repr__(self):
        s = ''
        s += f'calculator: {str(self.calculator.__class__):>60s}\n'
        s += f'filename:   {self.filename:>60s}\n'
        s += f'# Atoms:    {len(self.atoms):>60d}\n'
        s += f'# IDs:      {len(self.ids):>60d}'
        return s

