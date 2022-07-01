import numpy as np
from wfl.configset import OutputSpec, ConfigSet

import pytest

# tested modules
from user.generate import radicals
from wfl.generate import smiles

# wfl.generate_configs.smiles depends on rdkit.Chem
pytest.importorskip("rdkit.Chem")

def test_abstract_sp3_hydrogens():
    
    smiles_list = ['C', 'C=CCO']
    ConfigSet(input_configs=[smiles.smi_to_atoms(smi) for smi in smiles_list])
    cfs_out = OutputSpec()

    expected_formuli = [['CH3']*4,
                         ['C3OH5']*2]

    expected_config_types = [['rad1', 'rad2', 'rad3', 'rad4'],
                             ['rad7', 'rad8']]

    for mol, ref_formuli, ref_cfg_types in zip(cfs_out.output_configs,
                                               expected_formuli,
                                               expected_config_types):

        rads = radicals.abstract_sp3_hydrogen_atoms(mol)
        formuli = [str(at.symbols) for at in rads]
        assert np.all(formuli == ref_formuli)
        config_types = [at.info['config_type'] for at in rads]
        assert np.all(config_types == ref_cfg_types)

