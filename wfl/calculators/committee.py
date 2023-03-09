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
        self._calibrating = False

        self.committee_calculators = committee_calculators

        super().__init__(atoms=atoms)

    def calculate(self, atoms=None, properties=['energy', 'forces', 'stress'], system_changes=all_changes):
        """Calculates committee (mean) values and variances."""

        super().calculate(atoms, properties, system_changes)

        property_committee = {k_i: [] for k_i in properties}

        for cc_i in self.committee_calculators:
            cc_i.calculate(atoms=atoms, properties=properties, system_changes=system_changes)
            for p_i in properties:
                property_committee[p_i].append(cc_i.results[p_i])

        for p_i in properties:
            self.results[p_i] = np.mean(property_committee[p_i], axis=0)
            if self._calibrating:
                print(property_committee[p_i])
            self.results[f'{p_i}_uncertainty'] = np.sqrt(np.var(property_committee[p_i], ddof=1, axis=0))
            if p_i in self._alphas:
                self.results[f'{p_i}_uncertainty'] *= self._alphas[p_i]
            elif not self._calibrating:
                warnings.warn(f'Uncertainty estimation of CommitteeUncertainty-instance has not been calibrated for {p_i}.')

    def calibrate(self, properties, keys_ref, committee_filenames, appearance_threshold, system_changes=all_changes):
        """
        Determine calibration factors alpha (linear-scaling only) for specified properties
        by internal validation set, which will be used to scale the uncertainty estimation.

        Parameter:
        ----------
        properties: list(str)
            Properties to determine calibration factor alpha for.
        appearance_threshold: int
            For a sample to be selected as part of the validation set, it defines the maximum number
            this sample is allowed to appear in subsampled committee training sets.

        Returns:
        --------
        alphas: dict
            Stores calibration factors alpha (values) for specified properties (keys).
        """
        self._calibrating = True
        assert len(properties) == len(keys_ref)

        validation_set = self._get_internal_validation_set(committee_filenames, appearance_threshold)
        values_ref = {p_i: np.array([atoms_i.info[k_i] for atoms_i in validation_set]) for p_i, k_i in zip(properties, keys_ref)}

        values_pred = {p_i: [] for p_i in properties}
        var_pred = {f'{p_i}_variance': [] for p_i in properties}
        for atoms_i in validation_set:
            self.calculate(atoms=atoms_i, properties=properties, system_changes=system_changes)
            for p_i in properties:
                values_pred[p_i].append(self.results[p_i])
                var_pred[f'{p_i}_variance'].append(np.power(self.results[f'{p_i}_uncertainty'], 2))
        values_pred = {p_i: np.array(values_pred[p_i]) for p_i in properties}
        var_pred = {f'{p_i}_variance': np.array(var_pred[f'{p_i}_variance']) for p_i in properties}

        self._alphas.update({p_i: self._get_alpha(
                                    vals_ref=values_ref[p_i],
                                    vals_pred=values_pred[p_i],
                                    var_pred=var_pred[f'{p_i}_variance'],
                                    M=len(committee_filenames)
                                    )
                                  for p_i in properties
                            })
        self._calibrating = False

    def _get_internal_validation_set(self, committee_filenames, appearance_threshold):
        "Return samples found in `committee_filenames` that appear max. `appearance_threshold`-times."
        combined_subsets = [atoms_i for path_i in committee_filenames for atoms_i in ase.io.read(path_i, ':')]

        counter = Counter([atoms_i.info['_ConfigSet_loc__FullTraining'] for atoms_i in combined_subsets])
        print(counter)
        map_id_atoms = {atoms_i.info['_ConfigSet_loc__FullTraining']: atoms_i for atoms_i in combined_subsets}

        return [map_id_atoms[id_i] for id_i, count_i in counter.most_common() if count_i <= appearance_threshold]

    def _get_alpha(self, vals_ref, vals_pred, var_pred, M):
        "Get scaling factor alpha."
        N_val = len(vals_ref)
        print(var_pred)
        alpha_squared = -1/M + (M - 3)/(M - 1) * 1/N_val * np.sum(np.power(vals_ref-vals_pred, 2) / var_pred)
        return np.sqrt(alpha_squared)

