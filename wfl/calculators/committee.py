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
    deviation) is calculated.

    The idea for this Calculator class is based on the following publication:
    Musil et al., J. Chem. Theory Comput. 15, 906−915 (2019)
    https://pubs.acs.org/doi/full/10.1021/acs.jctc.8b00959

    Parameter:
    ----------
    committee_calculators: list(N)
        Collection of Calculators representing the committee.
    atoms : ase-Atoms
        Optional object to which the calculator will be attached.
    """

    def __init__(self, committee_calculators, atoms=None):

        self.implemented_properties = ['energy', 'forces', 'stress']

        # TODO: I tend to change the input argument to being a Committee-class instance
        # (instead of a list of calculators).
        # Related to this, I am also thinking to remove be calibrate() function below.
        # So, setting up the Committee and performing calculations with it become clearly
        # separated tasks.
        self.committee = Committee(
                members=[CommitteeMember(c_i) for c_i in committee_calculators]
                )

        super().__init__(atoms=atoms)

    def calibrate(self, properties, keys_ref, committee_filenames, appearance_threshold, system_changes=all_changes):
        """
        Calibrate the Uncertainty predictions of the committee.

        Parameter:
        ----------
        properties: list(M)
            Collection of strings specifying the properties for which uncertainty predictions
            will be calibrated.
        keys_ref: list(M)
            For each of the passed ```properties``` these strings represent the key
            under which the true value for that property is stored in a given sample
            (i.e. in ```committee_filenames```)
        appearance_threshold: int
            Number of times a sample for the validation set
            is maximally allowed to appear in the training set
            of a committee member.
        committee_filenames: list(N)
            Collection of paths to subsampled sets for training committee Calculators.
        """
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
                self.results[f'{p_i}_uncertainty'] = self.committee.scale_uncertainty(self.results[f'{p_i}_uncertainty'], p_i)
            else:
                warnings.warn(f'Uncertainty estimation has not been calibrated for {p_i}.')


class Committee:
    """
    Instances of this class represent a committee of models.

    It's use is to store the ```CommitteeMembers``` representing the committee model
    and to calibrate the obtained uncertainties (required when sub-sampling is used
    to create the training data of the committee members).

    Parameter:
    ----------
    member: list(M)
        List of ```CommitteeMember``` instances representing the committee model.
    """
    def __init__(self, members=[]):
        self.members = members
        self._update()

    @property
    def number(self):
        """Number of committee members."""
        return self._number

    @property
    def atoms(self):
        """Combined Atoms/samples in the committee."""
        return self._atoms

    @property
    def ids(self):
        """Identifiers of atoms/samples in the committee."""
        return self._ids

    @property
    def id_to_atoms(self):
        """Dictionary to translate identifiers to Atoms-objects."""
        return self._id_to_atoms

    @property
    def id_counter(self):
        """Counter-object for identifier appearances in the committee."""
        return self._id_counter

    @property
    def alphas(self):
        """(Linear) scaling factors for committee uncertainties."""
        return self._alphas

    @property
    def calibrated_for(self):
        """Set of properties the committee has been calibrated for."""
        return self._calibrated_for

    @property
    def validation_set(self):
        """List of Atoms-objects."""
        if self._validation_set:
            return self._validation_set
        else:
            raise AttributeError('`Committee`-instance has been altered since last call of `Committee.set_internal_validation_set()`.')

    def _update(self):
        self._number = len(self.members)
        self._atoms = [atoms_ij for cm_i in self.members for atoms_ij in cm_i.atoms]
        self._ids = [id_ij for cm_i in self.members for id_ij in cm_i.ids]
        self._id_to_atoms = {id_i: atoms_i for id_i, atoms_i in zip(self.ids, self.atoms)}
        self._id_counter = Counter(self.ids)
        self._validation_set = []
        self._alphas = {}
        self._calibrated_for = set()

    def is_calibrated_for(self, prop):
        """Check whether committee has been calibrated for ```prop```."""
        return prop in self._calibrated_for

    def __add__(self, member):
        """Extend committee by new ```member``` (i.e. CommitteeMember-instance)."""
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
        s += f'# atoms validation set:       {len(self._validation_set):>10d}\n'
        s += f'calibrated for:\n'
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
        """
        Read in and store the training data of the committee members from the passed ```filenames```.

        Parameter:
        ----------
        filenames: list(str)
            Paths to the training data of the committee members.
            Sorted in accordance with the committee members of the class-instance.
        """
        assert len(filenames) == self.number

        for cm_i, f_i in zip(self.members, filenames):
            cm_i.add_training_data(f_i)
        self._update()

    def set_internal_validation_set(self, appearance_threshold):
        """
        Define a validation set based on the Atoms-objects of subsampled committee training sets.

        appearance_threshold: int
            Number of times a sample for the validation set
            is maximally allowed to appear in the training set
            of a committee member.
        """

        assert appearance_threshold <= self.number - 2

        self._validation_set = []
        for id_i, appearance_i in self.id_counter.most_common()[::-1]:
            if appearance_i <= appearance_threshold:
                break
            self._validation_set.append(self.id_to_atoms[id_i])

    def calibrate(self, properties, keys_ref, system_changes=all_changes):
        """
        Obtain parameters to properly scale committee uncertainties and make
        them available as an attribute (```alphas```) with another associated
        attribute (```calibrated_for```) providing information about the property
        for which the uncertainty will be scales by it.

        properties: list(str)
            Properties for which the calibration will determine scaling factors.
        keys_ref: list(str)
            Keys under which the reference values in the validation set are stored
            (i.e. under Atoms.info[```keys_ref```]).
        """
        # TODO: read in calibration stored on disk (avoid recalibrating)
        # TODO: extend calibration for Atoms.arrays-properties.

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

            for p_i in properties:
                validation_pred[p_i][idx_i] = np.mean(sample_committee_pred[p_i])
                validation_pred_var[f'{p_i}_variance'][idx_i] = np.var(sample_committee_pred[p_i], ddof=1, axis=0)

        validation_ref = {p_i: np.array([atoms_i.info[k_i] for atoms_i in self.validation_set]) for p_i, k_i in zip(properties, keys_ref)}

        for p_i in properties:
            self._alphas.update(
                    {p_i: self._get_alpha(vals_ref=validation_ref[p_i],
                                          vals_pred=validation_pred[p_i],
                                          var_pred=validation_pred_var[f'{p_i}_variance'],
                                          )
                    })
            self._calibrated_for.add(p_i)

    def _get_alpha(self, vals_ref, vals_pred, var_pred):
        """
        Get (linear) uncertainty scaling factor alpha.

        This implementation is based on:
        Imbalzano et al., J. Chem. Phys. 154, 074102 (2021)
        https://doi.org/10.1063/5.0036522

        Parameter:
        ----------
        vals_ref: ndarray(N)
            Reference values for validation set samples.
        vals_pred: ndarray(N)
            Values predicted by the committee for validation set samples.
        var_pred: ndarray(N)
            Variance predicted by the committee for validation set samples.

        Returns:
        --------
        (Linear) uncertainty scaling factor alpha.
        """
        N_val = len(vals_ref)
        M = self.number
        alpha_squared = -1/M + (M - 3)/(M - 1) * 1/N_val * np.sum(np.power(vals_ref-vals_pred, 2) / var_pred)
        assert alpha_squared > 0, f'Obtained negative value for `alpha_squared`: {alpha_squared}'
        return np.sqrt(alpha_squared)

    def scale_uncertainty(self, value, prop):
        """Scale uncertainty ```value``` obtained with the committee according to the calibration for the corresponding property (```prop```)."""
        # TODO: generalize scaling as sigma_scaled = alpha * sigma**(gamma/2 + 1)
        return self.alphas[prop] * value


class CommitteeMember:
    """
    Lightweight class defining a member (i.e. a sub-model) of a committee model.

    Parameter:
    ----------
    calculator: Calculator
        Instance of a Calculator-class (or heirs e.g. quippy.potential.Potential)
        representing a machine-learned model.
    filename: str, optional default=None
        Path to the (sub-sampled) training set used to create the machine-learned model
        defined by the ```calculator```.
    """
    def __init__(self, calculator, filename=None):  # TODO: Allow both `filename` and list(Atoms) and/or ConfigSet?
        self._calculator = calculator
        self._filename = filename

        if self._filename is None:
            self._atoms = []
            self._ids = []
        else:
            self._update()

    @property
    def calculator(self):
        """Model of the committee member."""
        return self._calculator

    @property
    def filename(self):
        """Path to the atoms/samples in the committee member."""
        return self._filename

    @property
    def atoms(self):
        """Atoms/samples in the committee member."""
        return self._atoms

    @property
    def ids(self):
        """Identifiers of atoms/samples in the committee member."""
        return self._ids

    def _update(self):
        self._atoms = ase.io.read(self.filename, ':')
        self._ids = [atoms_i.info['_ConfigSet_loc__FullTraining'] for atoms_i in self.atoms]

    def add_training_data(self, filename):
        """
        Read in and store the training data of this committee members from the passed ```filename```.

        Parameter:
        ----------
        filename: str
            Path to the training data of the committee member.
        """
        self._filename = filename
        self._update()

    def is_sample_in_atoms(self, sample):
        """Check if passed Atoms-object is part of this committee member (by comparing identifiers)."""
        return sample.info['_ConfigSet_loc__FullTraining'] in self.ids

    def __repr__(self):
        s = ''
        s += f'calculator: {str(self.calculator.__class__):>60s}\n'
        s += f'filename:   {self.filename:>60s}\n'
        s += f'# Atoms:    {len(self.atoms):>60d}\n'
        s += f'# IDs:      {len(self.ids):>60d}'
        return s

